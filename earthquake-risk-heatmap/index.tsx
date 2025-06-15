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

interface ActualQuake {
    latitude: number;
    longitude: number;
    magnitude: number;
    depth: number;
}

interface ApiResponse {
    update_time: string;
    forecasts: ForecastItem[];
    actual_quakes: ActualQuake[];
}

// --- Configuration ---
const API_BASE_URL = '/api/predict_at'; // New API endpoint
const MAP_CENTER_JAPAN: L.LatLngTuple = [36.2048, 138.2529];
const INITIAL_ZOOM_LEVEL = 5;

// --- Global Variables ---
let mapInstance: L.Map | null = null;
let updateTimeElement: HTMLElement | null = null;
let yearSelectElement: HTMLSelectElement | null = null;
let monthSelectElement: HTMLSelectElement | null = null;
let predictButtonElement: HTMLButtonElement | null = null;

let gridLayerGroup: L.LayerGroup | null = null;
let actualQuakesLayerGroup: L.LayerGroup | null = null;
let legendControl: L.Control | null = null;

// --- Helper Functions ---

/**
 * Determines the color for a risk grid cell based on risk probability.
 */
function getRiskColor(probability: number): string {
    if (probability > 80) return '#800026';
    if (probability > 60) return '#BD0026';
    if (probability > 40) return '#E31A1C';
    if (probability > 20) return '#FED976';
    return '#ADFF2F';
}

/**
 * Determines the color for an actual quake marker based on depth.
 */
function getActualQuakeColor(depth: number): string {
    if (depth < 10) return '#FFFF00'; // Yellow (shallow)
    if (depth < 30) return '#FFA500'; // Orange
    if (depth < 70) return '#FF4500'; // OrangeRed
    return '#FF0000';    // Red (deep)
}

/**
 * Determines the radius for an actual quake marker based on magnitude.
 */
function getActualQuakeRadius(magnitude: number): number {
    return Math.max(4, magnitude * 2.5); // Min radius 4
}


/**
 * Populates the year select dropdown.
 * @param defaultYear - The year to be selected by default.
 */
function populateYearSelect(defaultYear: number): void {
    if (!yearSelectElement) return;
    const currentYear = new Date().getFullYear();
    for (let year = currentYear; year >= 2000; year--) {
        const option = document.createElement('option');
        option.value = String(year);
        option.textContent = `${year}년`;
        if (year === defaultYear) {
            option.selected = true;
        }
        yearSelectElement.appendChild(option);
    }
}

/**
 * Populates the month select dropdown.
 * @param defaultMonth - The month (1-12) to be selected by default.
 */
function populateMonthSelect(defaultMonth: number): void {
    if (!monthSelectElement) return;
    for (let month = 1; month <= 12; month++) {
        const option = document.createElement('option');
        option.value = String(month).padStart(2, '0');
        option.textContent = `${month}월`;
        if (month === defaultMonth) {
            option.selected = true;
        }
        monthSelectElement.appendChild(option);
    }
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
 * Adds a legend for risk probability to the map.
 */
function addRiskLegend(): void {
    if (!mapInstance) return;

    if (legendControl) {
        mapInstance.removeControl(legendControl);
    }
    legendControl = new L.Control({ position: 'bottomright' });
    legendControl.onAdd = function () {
        const div = L.DomUtil.create('div', 'info legend legend-risk');
        const gradesInfo = [
            { probability: '> 80%', color: getRiskColor(81) },
            { probability: '60% - 80%', color: getRiskColor(61) },
            { probability: '40% - 60%', color: getRiskColor(41) },
            { probability: '20% - 40%', color: getRiskColor(21) },
            { probability: '<= 20%', color: getRiskColor(19) }
        ];
        div.innerHTML = '<h4>AI 예측 위험도</h4>';
        for (const item of gradesInfo) {
            div.innerHTML += `<i style="background:${item.color}"></i> ${item.probability}<br>`;
        }
        return div;
    };
    legendControl.addTo(mapInstance);
}


/**
 * Fetches forecast and actual quake data from the backend API for a specific date.
 * @param dateString - The date in YYYY-MM format.
 * @returns A Promise resolving to ApiResponse or null if an error occurs.
 */
async function fetchPredictionData(dateString: string): Promise<ApiResponse | null> {
    const apiUrlWithDate = `${API_BASE_URL}?date=${dateString}`;
    try {
        const response = await fetch(apiUrlWithDate);
        if (!response.ok) {
            console.error(`API Error: ${response.status} - ${response.statusText} (URL: ${apiUrlWithDate})`);
            throw new Error(`Failed to fetch data for ${dateString}: HTTP ${response.status}`);
        }
        const data: ApiResponse = await response.json();

        if (updateTimeElement) {
            const date = new Date(data.update_time);
            const formattedTime = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
            updateTimeElement.textContent = `데이터 기준 시각: ${formattedTime}`;
        }
        return data;
    } catch (error) {
        console.error('Error fetching prediction data:', error);
        if (updateTimeElement) {
            updateTimeElement.textContent = `데이터 로딩 오류 (${dateString}).`;
        }
        return null;
    }
}

/**
 * Updates the choropleth map layers with new forecast data.
 * @param forecasts - An array of forecast items.
 */
function updateChoroplethMap(forecasts: ForecastItem[]): void {
    if (!mapInstance) return;

    if (gridLayerGroup) {
        mapInstance.removeLayer(gridLayerGroup);
        gridLayerGroup.clearLayers();
    } else {
        gridLayerGroup = L.layerGroup();
    }

    forecasts.forEach(item => {
        const rectangleBounds: L.LatLngBoundsExpression = item.bounds;
        const rectangle = L.rectangle(rectangleBounds, {
            fillColor: getRiskColor(item.risk_probability),
            fillOpacity: 0.7,
            weight: 0,
            interactive: true
        });
        const popupContent = `
            <b>격자 ID:</b> ${item.grid_id}<br>
            <b>AI 예측 위험 확률:</b> ${item.risk_probability.toFixed(1)}%<hr>
            <b>평균 규모 (해당 월):</b> ${item.details.mean.toFixed(2)}<br>
            <b>월평균 발생 횟수 (이전 12개월):</b> ${item.details.count_12m_avg.toFixed(2)}<br>
            <b>b-value:</b> ${item.details.b_value.toFixed(2)}<br>
            <b>이전 주요 지진 후 경과일:</b> ${item.details.days_since_last_quake}일
        `;
        rectangle.bindPopup(popupContent);
        gridLayerGroup.addLayer(rectangle);
    });
    gridLayerGroup.addTo(mapInstance);
}

/**
 * Updates the map layer with actual earthquake data.
 * @param quakes - An array of actual earthquake items.
 */
function updateActualQuakesLayer(quakes: ActualQuake[]): void {
    if (!mapInstance) return;

    if (actualQuakesLayerGroup) {
        mapInstance.removeLayer(actualQuakesLayerGroup);
        actualQuakesLayerGroup.clearLayers();
    } else {
        actualQuakesLayerGroup = L.layerGroup();
    }

    quakes.forEach(quake => {
        const marker = L.circleMarker([quake.latitude, quake.longitude], {
            radius: getActualQuakeRadius(quake.magnitude),
            fillColor: getActualQuakeColor(quake.depth),
            color: '#000', // Border color
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8,
            interactive: true
        });
        const popupContent = `
            <b>실제 발생 지진</b><hr>
            <b>규모:</b> ${quake.magnitude.toFixed(1)}<br>
            <b>깊이:</b> ${quake.depth.toFixed(0)} km
        `;
        marker.bindPopup(popupContent);
        actualQuakesLayerGroup.addLayer(marker);
    });
    actualQuakesLayerGroup.addTo(mapInstance);
}


/**
 * Main application routine to fetch data and update the map for a given date.
 * @param dateString - The date in YYYY-MM format.
 */
async function refreshMapData(dateString: string): Promise<void> {
    if (predictButtonElement) {
        predictButtonElement.disabled = true;
        predictButtonElement.textContent = '로딩 중...';
    }

    const apiResponse = await fetchPredictionData(dateString);
    if (apiResponse) {
        if (apiResponse.forecasts) {
            updateChoroplethMap(apiResponse.forecasts);
        }
        if (apiResponse.actual_quakes) {
            updateActualQuakesLayer(apiResponse.actual_quakes);
        }
    } else {
        // Clear layers if data fetch fails for the selected period
        if (gridLayerGroup) gridLayerGroup.clearLayers();
        if (actualQuakesLayerGroup) actualQuakesLayerGroup.clearLayers();
    }

    if (predictButtonElement) {
        predictButtonElement.disabled = false;
        predictButtonElement.textContent = '해당 시점 예측 보기';
    }
}

// --- Application Entry Point ---
document.addEventListener('DOMContentLoaded', () => {
    updateTimeElement = document.getElementById('update-time') as HTMLElement;
    yearSelectElement = document.getElementById('year-select') as HTMLSelectElement;
    monthSelectElement = document.getElementById('month-select') as HTMLSelectElement;
    predictButtonElement = document.getElementById('predict-button') as HTMLButtonElement;

    initializeMap();
    if (mapInstance) {
        addRiskLegend();
    }

    // Determine initial date (one month ago)
    const oneMonthAgo = new Date();
    oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
    const initialYear = oneMonthAgo.getFullYear();
    const initialMonth = oneMonthAgo.getMonth() + 1; // 1-12

    populateYearSelect(initialYear);
    populateMonthSelect(initialMonth);

    // Initial data load for one month ago
    const initialDateString = `${initialYear}-${String(initialMonth).padStart(2, '0')}`;
    refreshMapData(initialDateString);

    // Event listener for the predict button
    if (predictButtonElement) {
        predictButtonElement.addEventListener('click', () => {
            if (yearSelectElement && monthSelectElement) {
                const selectedYear = yearSelectElement.value;
                const selectedMonth = monthSelectElement.value;
                const dateString = `${selectedYear}-${selectedMonth}`;
                refreshMapData(dateString);
            }
        });
    }
});