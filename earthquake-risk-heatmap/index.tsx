
import L from 'leaflet';
import 'leaflet/dist/leaflet.css'; // Import Leaflet's CSS
import './index.css';   // Import custom styles

// --- Interfaces for API Data ---
interface ForecastDetails {
    count: number;
    mean: number;
    count_12m_avg: number;
    b_value: number;
    days_since_last_quake: number;
}

interface ForecastItem {
    grid_id: string;
    bounds: [[number, number], [number, number]]; // [[south, west], [north, east]]
    risk_probability: number;
    details: ForecastDetails;
}

interface ApiResponse {
    update_time: string;
    forecasts: ForecastItem[];
}

// --- Configuration ---
const API_URL = '/api/forecast';
const UPDATE_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes in milliseconds
const MAP_CENTER_JAPAN: L.LatLngTuple = [36.2048, 138.2529]; // Latitude, Longitude for central Japan
const INITIAL_ZOOM_LEVEL = 5;
// GRID_CELL_SIZE_DEGREES is no longer needed as bounds are provided by the API

// --- Global Variables ---
let mapInstance: L.Map | null = null;
let updateTimeElement: HTMLElement | null = null;
let gridLayerGroup: L.LayerGroup | null = null;
let legendControl: L.Control | null = null;

// --- Helper Functions ---

/**
 * Determines the color for a grid cell based on risk probability.
 * @param probability - The risk probability (0-100).
 * @returns A string representing the color code.
 */
function getColor(probability: number): string {
    if (probability > 80) return '#800026'; // 진한 빨강
    if (probability > 60) return '#BD0026'; // 빨강
    if (probability > 40) return '#E31A1C'; // 주황
    if (probability > 20) return '#FED976'; // 노랑
    return '#ADFF2F'; // 연두
}

/**
 * Initializes the Leaflet map.
 */
function initializeMap(): void {
    if (document.getElementById('map') && !mapInstance) {
        mapInstance = L.map('map').setView(MAP_CENTER_JAPAN, INITIAL_ZOOM_LEVEL);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19,
        }).addTo(mapInstance);
    } else if (!document.getElementById('map')) {
        console.error('Map container element #map not found.');
    }
}

/**
 * Adds a legend to the map.
 */
function addLegend(): void {
    if (!mapInstance) return;

    if (legendControl) {
        mapInstance.removeControl(legendControl);
    }

    legendControl = new L.Control({ position: 'bottomright' });

    legendControl.onAdd = function () {
        const div = L.DomUtil.create('div', 'info legend');
        const gradesInfo = [
            { probability: '> 80%', color: '#800026' },
            { probability: '60% - 80%', color: '#BD0026' },
            { probability: '40% - 60%', color: '#E31A1C' },
            { probability: '20% - 40%', color: '#FED976' },
            { probability: '<= 20%', color: '#ADFF2F' }
        ];

        div.innerHTML = '<h4>위험도 등급</h4>';
        for (const item of gradesInfo) {
            div.innerHTML +=
                '<i style="background:' + item.color + '"></i> ' +
                item.probability + '<br>';
        }
        return div;
    };
    legendControl.addTo(mapInstance);
}

/**
 * Fetches forecast data from the backend API.
 * @returns A Promise resolving to ApiResponse or null if an error occurs.
 */
async function fetchForecastData(): Promise<ApiResponse | null> {
    try {
        const response = await fetch(API_URL);
        if (!response.ok) {
            console.error(`API Error: ${response.status} - ${response.statusText}`);
            throw new Error(`Failed to fetch data: HTTP ${response.status}`);
        }
        const data: ApiResponse = await response.json();

        if (updateTimeElement) {
            const date = new Date(data.update_time);
            // Format date to be more human-readable, e.g. YYYY-MM-DD HH:mm:ss
             const formattedTime = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
            updateTimeElement.textContent = `마지막 업데이트: ${formattedTime}`;
        }
        return data;
    } catch (error) {
        console.error('Error fetching forecast data:', error);
        if (updateTimeElement) {
            updateTimeElement.textContent = '데이터 로딩 오류. 재시도 중...';
        }
        return null;
    }
}

/**
 * Updates the choropleth map layers with new forecast data.
 * @param forecasts - An array of forecast items.
 */
function updateChoroplethMap(forecasts: ForecastItem[]): void {
    if (!mapInstance) {
        console.warn('Map instance not available for choropleth update.');
        return;
    }

    if (gridLayerGroup) {
        mapInstance.removeLayer(gridLayerGroup);
        gridLayerGroup.clearLayers();
    }
    gridLayerGroup = L.layerGroup();

    forecasts.forEach(item => {
        // Use the bounds directly from the API response
        const rectangleBounds: L.LatLngBoundsExpression = item.bounds;

        const rectangle = L.rectangle(rectangleBounds, {
            fillColor: getColor(item.risk_probability),
            fillOpacity: 0.7,
            weight: 0, // Transparent border
            interactive: true // Ensure it can receive events
        });

        const popupContent = `
            <b>격자 ID:</b> ${item.grid_id}<br>
            <b>위험 확률:</b> ${item.risk_probability.toFixed(1)}%<hr>
            <b>평균 규모 (1개월):</b> ${item.details.mean.toFixed(2)}<br>
            <b>월평균 발생 횟수 (12개월):</b> ${item.details.count_12m_avg.toFixed(2)}<br>
            <b>b-value:</b> ${item.details.b_value.toFixed(2)}<br>
            <b>마지막 지진 후 경과일:</b> ${item.details.days_since_last_quake}일
        `;

        rectangle.bindPopup(popupContent);
        gridLayerGroup.addLayer(rectangle);
    });

    gridLayerGroup.addTo(mapInstance);
}

/**
 * Main application routine to fetch data and update the map.
 */
async function refreshMapData(): Promise<void> {
    const apiResponse = await fetchForecastData();
    if (apiResponse && apiResponse.forecasts) {
        updateChoroplethMap(apiResponse.forecasts);
    }
}

// --- Application Entry Point ---
document.addEventListener('DOMContentLoaded', () => {
    updateTimeElement = document.getElementById('update-time');
    
    initializeMap();
    if (mapInstance) {
        addLegend();
    }
    
    refreshMapData(); // Initial data load

    setInterval(refreshMapData, UPDATE_INTERVAL_MS); // Set up auto-refresh
});
