# Stock App Deployment Guide

## Production Checklist

### Environment Configuration
- [ ] Set `OPENROUTER_API_KEY` in production environment
- [ ] Configure `API_CACHE_ENABLED=true` for production
- [ ] Set `API_CACHE_TTL=3600` (1 hour)
- [ ] Update CORS origins to production domain
- [ ] Enable rate limiting for API endpoints

### Database & Data
- [ ] Migrate stock data to production database
- [ ] Set up automated data refresh jobs
- [ ] Configure backup strategy for ML models
- [ ] Verify all 500 stocks are loaded

### Performance Optimizations
- [x] Service worker for offline capability
- [x] Lazy loading for chart components
- [x] API response caching (1-hour TTL)
- [x] Pagination for large datasets
- [ ] CDN configuration for static assets
- [ ] Gzip compression enabled

### Security
- [ ] API rate limiting enabled
- [ ] Input validation on all endpoints
- [ ] Sanitize user inputs
- [ ] HTTPS enforced
- [ ] Security headers configured

### Monitoring & Logging
- [ ] Set up application monitoring (e.g., Sentry)
- [ ] Configure error logging
- [ ] Set up performance monitoring
- [ ] API health check endpoint active
- [ ] Alert notifications configured

### Frontend Build
```bash
cd frontend
npm run build
# Output: frontend/dist
```

### Backend Deployment

#### Using Docker
```bash
# Build image
docker build -t stock-app:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_key \
  --name stock-app-backend \
  stock-app:latest
```

#### Using PM2
```bash
# Install PM2
npm install -g pm2

# Start application
pm2 start "uvicorn backend.main:app --host 0.0.0.0 --port 8000" --name stock-app

# Save PM2 config
pm2 save
pm2 startup
```

### Deployment Platforms

#### Vercel (Frontend + Serverless Backend)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

#### Railway / Render (Full Stack)
1. Connect GitHub repository
2. Set environment variables
3. Configure build command: `pip install -r requirements.txt`
4. Configure start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

#### AWS EC2
1. Launch Ubuntu instance (t2.medium recommended)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip nginx
pip3 install -r requirements.txt
```
3. Configure Nginx as reverse proxy
4. Set up SSL with Let's Encrypt

### Post-Deployment
- [ ] Test all API endpoints
- [ ] Verify ML model predictions
- [ ] Check service worker offline capability
- [ ] Monitor initial traffic
- [ ] Set up automated backups

### Performance Targets
- API response time: < 500ms (95th percentile)
- Page load time: < 3s
- Time to interactive: < 5s
- Lighthouse score: > 90

### Rollback Plan
1. Keep previous Docker image tagged
2. Database migration rollback scripts ready
3. DNS TTL set to 5 minutes for quick switching
4. Backup of last working ML models
