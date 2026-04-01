/* ══════════════════════════════════════════
   Disease Drone — Dashboard JS
   ══════════════════════════════════════════ */

const API = '/api';
const DISEASE_COLORS = {
    healthy: '#10b981',
    leaf_blight: '#ef4444',
    leaf_spot: '#f59e0b',
    rust: '#d97706',
    powdery_mildew: '#8b5cf6',
};

const DISEASE_LABELS = {
    healthy: 'Healthy',
    leaf_blight: 'Leaf Blight',
    leaf_spot: 'Leaf Spot',
    rust: 'Rust',
    powdery_mildew: 'Powdery Mildew',
};

// ── State ──
let map;
let drawControl;
let drawnItems;
let detectionMarkers = L.layerGroup();
let sprayZoneLayer = L.layerGroup();
let healthHeatLayer = null;
let scanAreaLayer = L.layerGroup();
let currentTab = 'detections';
let mapMode = 'view'; // 'view' | 'draw'

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    initTabs();
    loadAllData();
    setInterval(loadStats, 15000);
});

// ── Map Setup ──
function initMap() {
    map = L.map('map', {
        center: [12.9716, 77.5946],
        zoom: 15,
        zoomControl: false,
    });

    L.control.zoom({ position: 'bottomleft' }).addTo(map);

    // Dark-styled tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OSM &copy; CARTO',
        maxZoom: 20,
    }).addTo(map);

    // Remove the tile brightness filter since we're using a dark tile layer
    document.querySelector('.leaflet-tile-pane').style.filter = 'none';

    // Layer groups
    detectionMarkers.addTo(map);
    sprayZoneLayer.addTo(map);
    scanAreaLayer.addTo(map);

    // Draw control
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    drawControl = new L.Control.Draw({
        draw: {
            polygon: {
                shapeOptions: {
                    color: '#3b82f6',
                    fillColor: '#3b82f620',
                    weight: 2,
                },
                allowIntersection: false,
            },
            rectangle: {
                shapeOptions: {
                    color: '#3b82f6',
                    fillColor: '#3b82f620',
                    weight: 2,
                },
            },
            circle: false,
            circlemarker: false,
            marker: false,
            polyline: false,
        },
        edit: {
            featureGroup: drawnItems,
        },
    });

    map.on(L.Draw.Event.CREATED, (e) => {
        drawnItems.addLayer(e.layer);
        setMapMode('view');
        toast('Scan area defined', 'success');
    });
}

function setMapMode(mode) {
    mapMode = mode;
    const label = document.getElementById('mapModeLabel');
    const toolbar = document.getElementById('mapToolbar');
    if (mode === 'draw') {
        map.addControl(drawControl);
        label.textContent = 'DRAW SCAN AREA — use toolbar on the left';
        label.classList.add('visible');
        toolbar.classList.add('hidden');
    } else {
        try { map.removeControl(drawControl); } catch(e) {}
        label.classList.remove('visible');
        toolbar.classList.remove('hidden');
    }
}

function toggleDrawMode() {
    setMapMode(mapMode === 'draw' ? 'view' : 'draw');
}

// ── Tabs ──
function initTabs() {
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel-section').forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`section-${target}`).classList.add('active');
            currentTab = target;
        });
    });
}

// ── Data Loading ──
async function loadAllData() {
    await Promise.all([
        loadStats(),
        loadDetections(),
        loadSprayZones(),
        loadTreatments(),
        loadHealth(),
    ]);
}

async function loadStats() {
    try {
        const stats = await fetch(`${API}/stats`).then(r => r.json());
        animateValue('statMissions', stats.total_missions || 0);
        animateValue('statActive', stats.active_missions || 0);
        animateValue('statDetections', stats.total_detections || 0);
        animateValue('statPending', stats.pending_zones || 0);

        // Update legend counts
        const dist = stats.disease_distribution || {};
        Object.keys(DISEASE_COLORS).forEach(d => {
            const el = document.getElementById(`legend-count-${d}`);
            if (el) el.textContent = dist[d] || 0;
        });

        // Health bar
        const healthFill = document.getElementById('healthFill');
        const healthValue = document.getElementById('healthValue');
        if (stats.avg_health != null) {
            const pct = Math.round(stats.avg_health * 100);
            healthFill.style.width = pct + '%';
            healthFill.style.background = pct > 70 ? 'var(--success)' : pct > 40 ? 'var(--warning)' : 'var(--danger)';
            healthValue.textContent = pct + '%';
        }
    } catch(e) {
        console.error('Failed to load stats:', e);
    }
}

async function loadDetections() {
    try {
        const data = await fetch(`${API}/detections`).then(r => r.json());
        detectionMarkers.clearLayers();
        const container = document.getElementById('detectionList');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = `<div class="empty-state"><p>No detections yet.<br>Run a scan mission to start.</p></div>`;
            return;
        }

        data.forEach((det, i) => {
            // Map marker
            if (det.lat && det.lon) {
                const color = DISEASE_COLORS[det.class_name] || '#64748b';
                const marker = L.circleMarker([det.lat, det.lon], {
                    radius: 6 + det.confidence * 4,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.6,
                    weight: 1.5,
                }).bindPopup(`
                    <strong>${DISEASE_LABELS[det.class_name] || det.class_name}</strong><br>
                    Confidence: ${(det.confidence * 100).toFixed(1)}%<br>
                    <small>${det.lat.toFixed(5)}, ${det.lon.toFixed(5)}</small>
                `);
                detectionMarkers.addLayer(marker);
            }

            // Card
            const confClass = det.confidence > 0.8 ? 'high' : det.confidence > 0.6 ? 'medium' : 'low';
            const card = document.createElement('div');
            card.className = 'detection-card';
            card.style.animationDelay = `${i * 0.03}s`;
            card.innerHTML = `
                <div class="detection-header">
                    <div class="detection-disease">
                        <span class="legend-dot ${det.class_name}"></span>
                        ${DISEASE_LABELS[det.class_name] || det.class_name}
                    </div>
                    <span class="detection-conf ${confClass}">${(det.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="detection-meta">
                    ${det.lat ? `<span>${det.lat.toFixed(4)}, ${det.lon.toFixed(4)}</span>` : ''}
                    <span>${formatTime(det.detected_at)}</span>
                </div>
            `;
            if (det.lat && det.lon) {
                card.addEventListener('click', () => {
                    map.flyTo([det.lat, det.lon], 17, { duration: 0.8 });
                });
            }
            container.appendChild(card);
        });
    } catch(e) {
        console.error('Failed to load detections:', e);
    }
}

async function loadSprayZones() {
    try {
        const data = await fetch(`${API}/spray-zones`).then(r => r.json());
        sprayZoneLayer.clearLayers();
        const container = document.getElementById('zoneList');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = `<div class="empty-state"><p>No spray zones generated.</p></div>`;
            return;
        }

        data.forEach((zone, i) => {
            const color = DISEASE_COLORS[zone.disease_type] || '#64748b';

            // Draw polygon on map
            if (zone.geometry) {
                const geo = typeof zone.geometry === 'string' ? JSON.parse(zone.geometry) : zone.geometry;
                const polygon = L.polygon(geo, {
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.2,
                    weight: 2,
                    dashArray: zone.status === 'pending' ? '6 4' : null,
                }).bindPopup(`
                    <strong>Spray Zone #${zone.id}</strong><br>
                    Disease: ${DISEASE_LABELS[zone.disease_type] || zone.disease_type}<br>
                    Severity: ${(zone.severity * 100).toFixed(0)}%<br>
                    Status: ${zone.status}
                `);
                sprayZoneLayer.addLayer(polygon);
            }

            // Card
            const card = document.createElement('div');
            card.className = 'zone-card';
            card.innerHTML = `
                <div class="zone-header">
                    <div class="zone-disease" style="color: ${color}">
                        ${DISEASE_LABELS[zone.disease_type] || zone.disease_type || 'Unknown'}
                    </div>
                    <span class="badge badge-${zone.status}">${zone.status}</span>
                </div>
                <div class="detection-meta">
                    <span>Severity: ${(zone.severity * 100).toFixed(0)}%</span>
                    <span>${formatTime(zone.created_at)}</span>
                </div>
                ${zone.status === 'pending' ? `
                <div class="zone-actions">
                    <button class="btn btn-success btn-sm" onclick="approveZone(${zone.id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
                        Approve
                    </button>
                    <button class="btn btn-danger btn-sm" onclick="rejectZone(${zone.id})">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                        Reject
                    </button>
                </div>` : ''}
            `;
            if (zone.center_lat && zone.center_lon) {
                card.querySelector('.zone-header').style.cursor = 'pointer';
                card.querySelector('.zone-header').addEventListener('click', () => {
                    map.flyTo([zone.center_lat, zone.center_lon], 17, { duration: 0.8 });
                });
            }
            container.appendChild(card);
        });
    } catch(e) {
        console.error('Failed to load spray zones:', e);
    }
}

async function loadTreatments() {
    try {
        const data = await fetch(`${API}/treatments`).then(r => r.json());
        const container = document.getElementById('treatmentList');
        container.innerHTML = '';

        if (data.length === 0) {
            container.innerHTML = `<div class="empty-state"><p>No treatments logged.</p></div>`;
            return;
        }

        data.forEach(t => {
            const card = document.createElement('div');
            card.className = 'treatment-card';
            card.innerHTML = `
                <div class="treatment-header">
                    <span class="treatment-chemical">${t.chemical || 'Unknown'}</span>
                    <span class="badge badge-treated">${t.disease_type || 'treated'}</span>
                </div>
                <div class="treatment-meta">
                    Duration: ${t.spray_duration ? t.spray_duration.toFixed(1) + 's' : 'N/A'}
                    &nbsp;|&nbsp; ${formatTime(t.treated_at)}
                    &nbsp;|&nbsp; ${t.lat.toFixed(4)}, ${t.lon.toFixed(4)}
                </div>
            `;
            container.appendChild(card);
        });
    } catch(e) {
        console.error('Failed to load treatments:', e);
    }
}

async function loadHealth() {
    try {
        const data = await fetch(`${API}/health`).then(r => r.json());
        if (data.length === 0) return;

        // Remove existing heat layer
        if (healthHeatLayer) map.removeLayer(healthHeatLayer);

        // Build heatmap data: invert health score so disease areas are "hot"
        const points = data.map(p => [p.lat, p.lon, 1 - p.health_score]);

        healthHeatLayer = L.heatLayer(points, {
            radius: 30,
            blur: 20,
            maxZoom: 18,
            max: 1.0,
            gradient: {
                0.0: '#10b981',
                0.3: '#84cc16',
                0.5: '#f59e0b',
                0.7: '#ef4444',
                1.0: '#dc2626',
            },
        });
    } catch(e) {
        console.error('Failed to load health:', e);
    }
}

// ── Actions ──

async function startScanMission() {
    let scanArea = null;
    if (drawnItems.getLayers().length > 0) {
        scanArea = [];
        drawnItems.getLayers()[0].getLatLngs()[0].forEach(ll => {
            scanArea.push([ll.lat, ll.lng]);
        });
    }

    const res = await fetch(`${API}/missions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: 'scan', scan_area: scanArea }),
    }).then(r => r.json());

    toast(`Scan mission #${res.id} created`, 'success');
    loadStats();
}

async function approveZone(zoneId) {
    await fetch(`${API}/spray-zones/${zoneId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'approved' }),
    });
    toast(`Zone #${zoneId} approved`, 'success');
    loadSprayZones();
    loadStats();
}

async function rejectZone(zoneId) {
    await fetch(`${API}/spray-zones/${zoneId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'rejected' }),
    });
    toast(`Zone #${zoneId} rejected`, 'info');
    loadSprayZones();
    loadStats();
}

async function seedDemoData() {
    toast('Seeding demo data...', 'info');
    await fetch(`${API}/demo/seed`, { method: 'POST' });
    toast('Demo data loaded!', 'success');
    await loadAllData();

    // Fly to demo area
    setTimeout(() => {
        map.flyTo([12.9716, 77.5946], 15, { duration: 1.2 });
    }, 300);
}

async function clearAllData() {
    if (!confirm('Clear all data? This cannot be undone.')) return;
    await fetch(`${API}/demo/clear`, { method: 'POST' });
    toast('All data cleared', 'info');
    detectionMarkers.clearLayers();
    sprayZoneLayer.clearLayers();
    if (healthHeatLayer) { map.removeLayer(healthHeatLayer); healthHeatLayer = null; }
    drawnItems.clearLayers();
    await loadAllData();
}

function toggleHeatmap() {
    if (!healthHeatLayer) {
        toast('No health data available', 'info');
        return;
    }
    if (map.hasLayer(healthHeatLayer)) {
        map.removeLayer(healthHeatLayer);
        toast('Heatmap hidden', 'info');
    } else {
        healthHeatLayer.addTo(map);
        toast('Health heatmap enabled', 'success');
    }
}

function centerMap() {
    // Try to center on detection markers
    if (detectionMarkers.getLayers().length > 0) {
        map.fitBounds(detectionMarkers.getBounds(), { padding: [40, 40], maxZoom: 16 });
    }
}

// ── Utilities ──

function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts + 'Z');
    return d.toLocaleString(undefined, {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

function animateValue(elId, target) {
    const el = document.getElementById(elId);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;
    const diff = target - current;
    const steps = 20;
    const stepVal = diff / steps;
    let frame = 0;
    const anim = setInterval(() => {
        frame++;
        el.textContent = Math.round(current + stepVal * frame);
        if (frame >= steps) {
            el.textContent = target;
            clearInterval(anim);
        }
    }, 25);
}

function toast(message, type = 'info') {
    const container = document.getElementById('toasts');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = message;
    container.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}
