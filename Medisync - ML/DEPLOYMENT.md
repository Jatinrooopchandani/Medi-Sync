# Medisync ML Deployment Guide

This guide outlines the steps required to deploy the Medisync ML service in a production environment.

## Prerequisites

1. Linux server with:
   - Python 3.8 or higher
   - systemd
   - Sufficient disk space (minimum 10GB recommended)
   - At least 4GB RAM
   - Regular backup system

2. Create service user:
```bash
sudo useradd -r -s /bin/false medisync
sudo mkdir -p /opt/medisync-ml
sudo chown medisync:medisync /opt/medisync-ml
```

## Installation Steps

1. **Copy Application Files**
```bash
sudo cp -r ./* /opt/medisync-ml/
sudo chown -R medisync:medisync /opt/medisync-ml
```

2. **Set Up Virtual Environment**
```bash
cd /opt/medisync-ml
sudo -u medisync python3 -m venv venv
sudo -u medisync ./venv/bin/pip install -r requirements.txt
```

3. **Configure Environment**
```bash
sudo -u medisync cp .env.example .env
sudo -u medisync vim .env  # Edit configuration as needed
```

4. **Set Up System Service**
```bash
sudo cp medisync.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable medisync
```

5. **Set Up Log Rotation**
```bash
sudo mkdir -p /var/log/medisync
sudo chown medisync:medisync /var/log/medisync
sudo cp medisync.logrotate /etc/logrotate.d/medisync
```

6. **Configure Monitoring**
```bash
sudo cp medisync.crontab /etc/cron.d/medisync
sudo chown root:root /etc/cron.d/medisync
sudo chmod 644 /etc/cron.d/medisync
```

## Security Configuration

1. **Firewall Rules**
```bash
# Allow only necessary ports
sudo ufw allow 8080/tcp  # Application port
sudo ufw enable
```

2. **SSL/TLS Setup**
```bash
# If using Nginx as reverse proxy
sudo cp medisync.nginx.conf /etc/nginx/sites-available/medisync
sudo ln -s /etc/nginx/sites-available/medisync /etc/nginx/sites-enabled/
sudo certbot --nginx -d yourdomain.com
```

3. **File Permissions**
```bash
cd /opt/medisync-ml
sudo ./start.sh check-permissions
```

## Monitoring Setup

1. **System Checks**
```bash
# Test system check script
sudo -u medisync ./system_check.py

# Test monitoring script
sudo -u medisync ./monitor.py
```

2. **Configure Alerts**
```bash
# Edit monitoring configuration
sudo -u medisync vim monitoring_config.json
```

## Backup Configuration

1. **Set Up Backup Directory**
```bash
sudo mkdir -p /backup/medisync
sudo chown medisync:medisync /backup/medisync
```

2. **Test Backup Process**
```bash
sudo -u medisync ./start.sh backup
```

## Startup Verification

1. **Validate Installation**
```bash
sudo -u medisync ./start.sh validate
```

2. **Start Service**
```bash
sudo systemctl start medisync
sudo systemctl status medisync
```

3. **Verify Logs**
```bash
sudo journalctl -u medisync -f
```

## Health Check

1. **Verify Application Health**
```bash
curl http://localhost:8080/health
```

2. **Monitor Resource Usage**
```bash
sudo -u medisync ./monitor.py --check resources
```

## Production Checklist

- [ ] Debug mode disabled
- [ ] Secure configuration values set
- [ ] SSL/TLS configured
- [ ] Firewall rules active
- [ ] Monitoring alerts configured
- [ ] Backup system tested
- [ ] Log rotation configured
- [ ] System service enabled
- [ ] File permissions verified
- [ ] Security headers enabled

## Troubleshooting

1. **Service Won't Start**
   - Check logs: `journalctl -u medisync -n 50`
   - Verify permissions: `./start.sh check-permissions`
   - Validate dependencies: `./start.sh validate`

2. **Monitoring Alerts**
   - Check monitoring logs: `tail -f logs/monitoring.log`
   - Verify alert settings in monitoring_config.json

3. **Performance Issues**
   - Check system resources: `./monitor.py --check resources`
   - Review access logs
   - Check model loading times

## Maintenance

Regular maintenance tasks:
1. Check logs daily
2. Review monitoring reports
3. Test backups weekly
4. Update dependencies monthly
5. Rotate SSL certificates
6. Review system metrics

## Support

For production support:
1. Check application logs
2. Review monitoring reports
3. Run system diagnostics
4. Contact system administrator