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
    analyzed_month: string;
    forecasts: ForecastItem[];
    actual_quakes: ActualQuake[];
}

// --- Configuration ---
const API_BASE_URL_PREDICT_AT = '/api/predict_at';
const API_BASE_URL_LATEST = '/api/latest';
const MAP_CENTER_JAPAN: L.LatLngTuple = [36.2048, 138.2529];
const INITIAL_ZOOM_LEVEL = 5;

// --- Global Variables ---
let mapInstance: L.Map | null = null;
let analysisMonthElement: HTMLElement | null = null;
let monthPickerElement: HTMLInputElement | null = null;
let predictButtonElement: HTMLButtonElement | null = null;
let latestButtonElement: HTMLButtonElement | null = null; // New button element

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
    return '#ADFF2F'; // Light green for low risk
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
        const div = L.DomUtil.create('div', 'legend legend-risk');
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
 * Fetches and processes data from the backend API.
 * @param apiUrl - The full API URL to fetch data from.
 * @param contextDescription - A description of the data being fetched (e.g., "YYYY-MM" or "최신") for error messages.
 * @returns A Promise resolving to ApiResponse or null if an error occurs.
 */
async function fetchAndProcessApiData(apiUrl: string, contextDescription: string): Promise<ApiResponse | null> {
    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            const errorText = await response.text();
            console.error(`API Error: ${response.status} - ${response.statusText} (URL: ${apiUrl}). Body: ${errorText}`);
            throw new Error(`Failed to fetch data for ${contextDescription}: HTTP ${response.status}`);
        }
        const data: ApiResponse = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching or processing API data:', error);
        if (analysisMonthElement) {
            analysisMonthElement.textContent = `데이터 로딩 오류 (${contextDescription}).`;
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
 * Sets the loading state for UI elements, disabling buttons and showing status.
 * @param isLoading - True if loading, false otherwise.
 * @param actionSource - Optional: 'latest' or 'predict' to indicate which button triggered the load.
 */
function setLoadingState(isLoading: boolean, actionSource?: 'latest' | 'predict'): void {
    const defaultPredictText = '해당 시점 예측 보기';
    const defaultLatestText = '최신 예보 보기';
    const loadingText = '로딩 중...';

    if (predictButtonElement) {
        predictButtonElement.disabled = isLoading;
        if (isLoading && actionSource === 'predict') {
            predictButtonElement.textContent = loadingText;
        } else {
            predictButtonElement.textContent = defaultPredictText;
        }
    }
    if (latestButtonElement) {
        latestButtonElement.disabled = isLoading;
        if (isLoading && actionSource === 'latest') {
            latestButtonElement.textContent = loadingText;
        } else {
            latestButtonElement.textContent = defaultLatestText;
        }
    }
}

/**
 * Fetches data (either latest or for a specific date) and updates the map.
 * @param source - 'initial', 'latest', or 'predict'.
 * @param dateString - The date in YYYY-MM format, required if source is 'predict'.
 */
async function loadMapData(source: 'initial' | 'latest' | 'predict', dateString?: string): Promise<void> {
    let apiUrl: string;
    let contextDescription: string;
    let actionSource: 'latest' | 'predict' | undefined = undefined;

    if (source === 'initial' || source === 'latest') {
        apiUrl = API_BASE_URL_LATEST;
        contextDescription = '최신 예보';
        if (source === 'latest') actionSource = 'latest';
    } else if (source === 'predict') {
        if (!dateString) {
            if (analysisMonthElement) {
                analysisMonthElement.textContent = '분석 월을 선택해주세요.';
            }
            console.warn('Date string is required for "predict" source.');
            return;
        }
        apiUrl = `${API_BASE_URL_PREDICT_AT}?date=${dateString}`;
        contextDescription = dateString;
        actionSource = 'predict';
    } else {
        console.error('Invalid source for loadMapData:', source);
        return;
    }

    setLoadingState(true, actionSource);
    if (source === 'initial' && analysisMonthElement) {
        analysisMonthElement.textContent = '최신 데이터 로딩 중...';
    }

    const apiResponse = await fetchAndProcessApiData(apiUrl, contextDescription);

    if (apiResponse) {
        if (analysisMonthElement) {
            analysisMonthElement.textContent = `분석 대상 월: ${apiResponse.analyzed_month}`;
        }
        if (apiResponse.forecasts) {
            updateChoroplethMap(apiResponse.forecasts);
        }
        if (apiResponse.actual_quakes) {
            updateActualQuakesLayer(apiResponse.actual_quakes);
        }
    } else {
        // Error message is set by fetchAndProcessApiData within analysisMonthElement
        if (gridLayerGroup) gridLayerGroup.clearLayers();
        if (actualQuakesLayerGroup) actualQuakesLayerGroup.clearLayers();
         if (source === 'initial' && analysisMonthElement && !apiResponse) {
            // analysisMonthElement already updated with error by fetchAndProcessApiData
        }
    }
    setLoadingState(false, actionSource);
}

// --- Application Entry Point ---
document.addEventListener('DOMContentLoaded', () => {
    analysisMonthElement = document.getElementById('analysis-month') as HTMLElement;
    monthPickerElement = document.getElementById('month-picker') as HTMLInputElement;
    predictButtonElement = document.getElementById('predict-button') as HTMLButtonElement;
    latestButtonElement = document.getElementById('latest-button') as HTMLButtonElement; // Get the new button

    initializeMap();
    if (mapInstance) {
        addRiskLegend();
    }

    // Initial data load for the latest forecast
    loadMapData('initial');

    // Event listener for the '해당 시점 예측 보기' button
    if (predictButtonElement && monthPickerElement) {
        predictButtonElement.addEventListener('click', () => {
            const selectedDate = monthPickerElement!.value; // Value is YYYY-MM
            if (selectedDate) {
                loadMapData('predict', selectedDate);
            } else {
                if(analysisMonthElement) {
                    analysisMonthElement.textContent = '분석 월을 선택해주세요.';
                }
                console.warn('Month picker has no value selected for predict button.');
            }
        });
    }

    // Event listener for the '최신 예보 보기' button
    if (latestButtonElement) {
        latestButtonElement.addEventListener('click', () => {
            loadMapData('latest');
        });
    }
});
