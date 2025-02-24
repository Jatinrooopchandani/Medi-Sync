"""Secure model backup and recovery utilities"""
import os
import shutil
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import tarfile

logger = logging.getLogger(__name__)

class BackupError(Exception):
    """Raised when backup operations fail"""
    pass

class ModelBackupManager:
    """Secure model backup and recovery manager"""
    
    def __init__(self, 
                 models_dir: str,
                 backup_dir: Optional[str] = None,
                 max_backups: int = 5):
        """Initialize backup manager.
        
        Args:
            models_dir: Directory containing model files
            backup_dir: Directory for storing backups (default: models_dir/backups)
            max_backups: Maximum number of backups to keep
        """
        self.models_dir = Path(models_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.models_dir / 'backups'
        self.max_backups = max_backups
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Secure backup directory permissions
        os.chmod(self.backup_dir, 0o750)
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _create_backup_manifest(self, backup_files: List[Path]) -> Dict[str, Any]:
        """Create a manifest for the backup."""
        manifest = {
            'timestamp': datetime.utcnow().isoformat(),
            'files': {}
        }
        
        for filepath in backup_files:
            manifest['files'][filepath.name] = {
                'hash': self._compute_file_hash(filepath),
                'size': filepath.stat().st_size,
                'modified': datetime.fromtimestamp(
                    filepath.stat().st_mtime
                ).isoformat()
            }
            
        return manifest
    
    def create_backup(self) -> Dict[str, Any]:
        """Create a secure backup of all model files.
        
        Returns:
            Dict containing backup metadata
            
        Raises:
            BackupError: If backup creation fails
        """
        try:
            # Create timestamp for backup
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_name = f"model_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # Create temporary directory for staging
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy model files to staging
                model_files = []
                for file_path in self.models_dir.glob('*.{json,keras,h5}'):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        dest_path = temp_path / file_path.name
                        shutil.copy2(file_path, dest_path)
                        model_files.append(dest_path)
                
                # Create manifest
                manifest = self._create_backup_manifest(model_files)
                manifest_path = temp_path / 'backup_manifest.json'
                with open(manifest_path, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # Create secure archive
                archive_path = backup_path.with_suffix('.tar.gz')
                with tarfile.open(archive_path, 'w:gz') as tar:
                    tar.add(temp_path, arcname=backup_name)
                
                # Secure archive permissions
                os.chmod(archive_path, 0o440)
            
            # Clean old backups if needed
            self._cleanup_old_backups()
            
            logger.info(f"Created backup: {archive_path}")
            return {
                'backup_name': backup_name,
                'timestamp': manifest['timestamp'],
                'location': str(archive_path),
                'manifest': manifest
            }
            
        except Exception as e:
            raise BackupError(f"Failed to create backup: {str(e)}")
    
    def restore_backup(self, backup_name: str, validate: bool = True) -> Dict[str, Any]:
        """Restore models from a backup.
        
        Args:
            backup_name: Name of backup to restore
            validate: Whether to validate restored files
            
        Returns:
            Dict containing restore operation metadata
            
        Raises:
            BackupError: If restore operation fails
        """
        try:
            # Find backup archive
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            if not archive_path.exists():
                raise BackupError(f"Backup not found: {backup_name}")
            
            # Create temporary directory for restoration
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract archive
                with tarfile.open(archive_path, 'r:gz') as tar:
                    # Verify no absolute paths or parent directory references
                    for member in tar.getmembers():
                        if member.name.startswith('/') or '..' in member.name:
                            raise BackupError(f"Invalid path in archive: {member.name}")
                    tar.extractall(temp_path)
                
                # Load and verify manifest
                manifest_path = temp_path / backup_name / 'backup_manifest.json'
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Validate files if requested
                if validate:
                    for filename, info in manifest['files'].items():
                        file_path = temp_path / backup_name / filename
                        if not file_path.exists():
                            raise BackupError(f"Missing file in backup: {filename}")
                        
                        actual_hash = self._compute_file_hash(file_path)
                        if actual_hash != info['hash']:
                            raise BackupError(
                                f"Hash mismatch for {filename}: "
                                f"expected {info['hash']}, got {actual_hash}"
                            )
                
                # Create backup of current models before restore
                current_backup = self.create_backup()
                
                # Restore files to models directory
                for filename in manifest['files'].keys():
                    src_path = temp_path / backup_name / filename
                    dest_path = self.models_dir / filename
                    shutil.copy2(src_path, dest_path)
                    os.chmod(dest_path, 0o440)  # Read-only
            
            logger.info(f"Restored backup: {backup_name}")
            return {
                'backup_name': backup_name,
                'timestamp': datetime.utcnow().isoformat(),
                'restored_files': list(manifest['files'].keys()),
                'previous_backup': current_backup['backup_name']
            }
            
        except Exception as e:
            raise BackupError(f"Failed to restore backup: {str(e)}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups with metadata."""
        backups = []
        for archive in sorted(self.backup_dir.glob('*.tar.gz')):
            try:
                with tarfile.open(archive, 'r:gz') as tar:
                    backup_name = archive.stem.rsplit('.', 1)[0]
                    manifest_path = f"{backup_name}/backup_manifest.json"
                    manifest_file = tar.extractfile(manifest_path)
                    if manifest_file:
                        manifest = json.load(manifest_file)
                        backups.append({
                            'name': backup_name,
                            'timestamp': manifest['timestamp'],
                            'size': archive.stat().st_size,
                            'files': list(manifest['files'].keys())
                        })
            except Exception as e:
                logger.warning(f"Failed to read backup {archive.name}: {str(e)}")
        
        return backups
    
    def _cleanup_old_backups(self):
        """Remove old backups exceeding max_backups limit."""
        backups = sorted(
            self.backup_dir.glob('*.tar.gz'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for old_backup in backups[self.max_backups:]:
            try:
                old_backup.unlink()
                logger.info(f"Removed old backup: {old_backup.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup.name}: {str(e)}")
    
    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        """Verify integrity of a backup.
        
        Args:
            backup_name: Name of backup to verify
            
        Returns:
            Dict containing verification results
            
        Raises:
            BackupError: If verification fails
        """
        try:
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            if not archive_path.exists():
                raise BackupError(f"Backup not found: {backup_name}")
            
            verification_results = {
                'backup_name': backup_name,
                'timestamp': datetime.utcnow().isoformat(),
                'verified_files': [],
                'issues': []
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract and verify archive
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                
                # Load manifest
                manifest_path = temp_path / backup_name / 'backup_manifest.json'
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Verify each file
                for filename, info in manifest['files'].items():
                    file_path = temp_path / backup_name / filename
                    if not file_path.exists():
                        verification_results['issues'].append(
                            f"Missing file: {filename}"
                        )
                        continue
                    
                    actual_hash = self._compute_file_hash(file_path)
                    if actual_hash != info['hash']:
                        verification_results['issues'].append(
                            f"Hash mismatch for {filename}"
                        )
                    else:
                        verification_results['verified_files'].append(filename)
            
            verification_results['success'] = not verification_results['issues']
            return verification_results
            
        except Exception as e:
            raise BackupError(f"Failed to verify backup: {str(e)}")