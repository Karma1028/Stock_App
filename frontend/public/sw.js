"""
Service worker for offline capability and caching
"""

const CACHE_NAME = 'stock-app-v1';
const STATIC_CACHE = 'stock-app-static-v1';
const DYNAMIC_CACHE = 'stock-app-dynamic-v1';

// Assets to cache on install
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/src/main.tsx',
    '/src/index.css',
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    '/api/dashboard',
    '/api/stocks',
    '/api/news',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('[SW] Installing service worker...');

    event.waitUntil(
        caches.open(STATIC_CACHE).then((cache) => {
            console.log('[SW] Caching static assets');
            return cache.addAll(STATIC_ASSETS);
        })
    );

    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating service worker...');

    event.waitUntil(
        caches.keys().then((keys) => {
            return Promise.all(
                keys
                    .filter((key) => key !== STATIC_CACHE && key !== DYNAMIC_CACHE)
                    .map((key) => caches.delete(key))
            );
        })
    );

    return self.clients.claim();
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Skip cross-origin requests
    if (url.origin !== location.origin) {
        return;
    }

    // Cache strategy for API calls
    if (API_CACHE_PATTERNS.some(pattern => url.pathname.includes(pattern))) {
        event.respondWith(
            caches.open(DYNAMIC_CACHE).then((cache) => {
                return fetch(request)
                    .then((response) => {
                        // Clone and cache the response
                        cache.put(request, response.clone());
                        return response;
                    })
                    .catch(() => {
                        // Fallback to cache if network fails
                        return cache.match(request);
                    });
            })
        );
        return;
    }

    // Cache-first strategy for static assets
    event.respondWith(
        caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse;
            }

            return fetch(request).then((response) => {
                // Don't cache if not a successful response
                if (!response || response.status !== 200 || response.type === 'error') {
                    return response;
                }

                const responseToCache = response.clone();
                caches.open(DYNAMIC_CACHE).then((cache) => {
                    cache.put(request, responseToCache);
                });

                return response;
            });
        })
    );
});

// Message event - handle cache clearing
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'CLEAR_CACHE') {
        event.waitUntil(
            caches.keys().then((keys) => {
                return Promise.all(keys.map((key) => caches.delete(key)));
            })
        );
    }
});
