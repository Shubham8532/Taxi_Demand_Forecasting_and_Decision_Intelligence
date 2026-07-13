// Global variables
        let firstPrediction = true;
        let animationTimer = null;
        let routeLine = null;
        let taxiMarker = null;
        let map, markers = [];
        let allRegions = [];
        let selectedRegion = null;
        let trendChart = null;

        function loadingStep(text, percent) {
            document.getElementById("loadingStatus").textContent = text;
            document.getElementById("loadingBar").style.width = percent + "%";
        }

        const taxiIcon = L.divIcon({
                className: "",
                html: `<img id="taxi-car"
        src="/static/images/car.png"
        style="
            width:32px;
            height:32px;
            transition:transform .12s linear;
        ">`,
                iconSize: [60, 60],
                iconAnchor: [18, 28]
            });

        // Initialize map
        function initMap() {
            if (map) return;
            map = L.map('map').setView([40.7580, -73.9855], 11);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);
        }

        // Load regions
        async function loadRegions() {
            try {
                const response = await fetch('/get_regions');
                const data = await response.json();
                if (data.success) {
                    allRegions = data.regions;
                    populateZoneDropdown();
                    addMarkersToMap();
                }
            } catch (error) {
                console.error('Error loading regions:', error);
            }
        }

        // Populate zone dropdown
        function populateZoneDropdown() {
            const select = document.getElementById('zoneSelect');
            Object.entries(allRegions).forEach(([id, region]) => {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = region.name;
                select.appendChild(option);
            });
        }

        // Add centroid markers to map (before prediction)
        function addMarkersToMap() {
            Object.entries(allRegions).forEach(([id, region]) => {
                const marker = L.circleMarker([region.lat, region.lon], {
                    radius: 0,
                    fillColor: '#667eea',
                    color: '#fff',
                    weight: 2,
                    opacity: 0,
                    fillOpacity: 0
                }).addTo(map);

                marker.bindPopup(`<b>${region.name}</b><br>Zone ID: ${id}<br>Click to select`);
                marker.on('click', () => selectZone(id));
                markers.push(marker);
            });
        }

        // Add scatter points to map (after prediction)
        async function showScatterPoints() {
            try {
                const response = await fetch('/get_scatter_points');
                const data = await response.json();

                if (data.success) {
                    data.points.forEach(point => {
                        const dot = L.circleMarker([point.lat, point.lon], {
                            radius: 3,
                            fillColor: point.color,
                            color: point.color,
                            weight: 1,
                            opacity: 0.6,
                            fillOpacity: 0.6
                        }).addTo(map);
                        markers.push(dot);
                    });
                }
            } catch (error) {
                console.error('Error loading scatter points:', error);
            }
        }

        // Select zone
        function selectZone(zoneId) {
            selectedRegion = zoneId;
            const region = allRegions[zoneId];

            document.getElementById('zoneSelect').value = zoneId;
            document.getElementById('selectedZoneInfo').classList.remove('hidden');
            document.getElementById('selectedZoneName').textContent = region.name;
            document.getElementById('selectedZoneId').textContent = zoneId;

            checkPredictButton();
        }

        // Check if predict button should be enabled
        function checkPredictButton() {
            const zone = document.getElementById('zoneSelect').value;
            const date = document.getElementById('datePicker').value;
            const time = document.getElementById('timePicker').value;

            document.getElementById('predictBtn').disabled = !(zone && date && time);
        }

        // Set time
        function setTime(time) {
            document.getElementById('timePicker').value = time;
            checkPredictButton();
        }

        // Load available times
        async function loadAvailableTimes() {
            const date = document.getElementById('datePicker').value;
            if (!date) return;

            try {
                const response = await fetch('/get_available_times', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ date })
                });
                const data = await response.json();

                if (data.success) {

                    const select = document.getElementById('timePicker');
                    select.innerHTML = '<option value="">-- Select Time --</option>';

                    data.times.forEach(time => {
                        const option = document.createElement('option');
                        option.value = time;
                        option.textContent = time;
                        select.appendChild(option);
                    });

                    // Auto-select next 15-minute interval
                    const now = new Date();

                    const totalMinutes =
                        now.getHours() * 60 + now.getMinutes();

                    const nextQuarter =
                        Math.ceil(totalMinutes / 15) * 15;

                    const hh =
                        String(Math.floor(nextQuarter / 60)).padStart(2, "0");

                    const mm =
                        String(nextQuarter % 60).padStart(2, "0");

                    const nearestTime = `${hh}:${mm}`;

                    if (data.times.includes(nearestTime)) {
                        select.value = nearestTime;
                    }
                    else if (data.times.length > 0) {
                        select.value = data.times[0];
                    }

                    checkPredictButton();
                }
            } catch (error) {
                console.error('Error loading times:', error);
            }
        }

        // Make prediction
        async function makePrediction() {
            const region_id = document.getElementById('zoneSelect').value;
            const date = document.getElementById('datePicker').value;
            const time = document.getElementById('timePicker').value;

            if (!region_id || !date || !time) {
                alert('Please select zone, date, and time');
                return;
            }

            try {

                // Show loading card
                if (firstPrediction) {

                    document.getElementById("loadingCard").classList.remove("hidden");
                    document.getElementById("mainPredictionCard").classList.add("hidden");
                    document.getElementById("analyticsRow").classList.add("hidden");

                    loadingStep("Loading prediction model...", 20);

                }

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ region_id, date, time })
                });

                if (!response.ok) {
                    const text = await response.text();
                    alert("❌ Server error:\n" + text);

                    document.getElementById("loadingCard").classList.add("hidden");
                    return;
                }

                // Backend finished
                if (firstPrediction)
                    loadingStep("Finding best relocation route...", 60);

                const data = await response.json();

                if (data.success) {

                    if (firstPrediction)
                        loadingStep("Rendering dashboard...", 95);

                    displayResults(data);
                    if (firstPrediction) {
                        document.getElementById("loadingCard").classList.add("hidden");
                        firstPrediction = false;
                    }

                } else {

                    document.getElementById("loadingCard").classList.add("hidden");
                    alert("❌ Backend error:\n" + data.error);

                }

            } catch (error) {

                document.getElementById("loadingCard").classList.add("hidden");
                alert("❌ Fetch failed:\n" + error);

        }
        }

        // Display results
        function displayResults(data) {
            document.getElementById("loadingCard").classList.add("hidden");
            document.getElementById('resultsSection').classList.remove('hidden');
            document.getElementById('mainPredictionCard').classList.remove('hidden');
            document.getElementById('analyticsRow').classList.remove('hidden');
            document.getElementById('allZonesSection').classList.add('hidden');

            // Main prediction
            document.getElementById('predictedValue').textContent = data.prediction.predicted_demand;
            document.getElementById('nextInterval').textContent = data.prediction.next_interval;
            document.getElementById('confidenceLevel').textContent = data.prediction.confidence;
            if (data.recommendations.length > 0) {

                document.getElementById("etaValue").textContent =
                    data.recommendations[0].eta_minutes + " min";

                document.getElementById("gainValue").textContent =
                    "+" + data.recommendations[0].expected_gain;

            } else {

                document.getElementById("etaValue").textContent = "--";
                document.getElementById("gainValue").textContent = "--";

            }
            const trend = document.getElementById("trendIcon");
            if (trend) {
                trend.textContent = data.features.trend_icon;
            }

            // Accuracy
            document.getElementById('predictedDemand').textContent = data.prediction.predicted_demand;
            document.getElementById('actualDemand').textContent = data.prediction.actual_demand;
            document.getElementById('errorValue').textContent = `±${data.prediction.error} (${data.prediction.error_percent}%)`;

            // Features
            document.getElementById('dayOfWeek').textContent = data.features.day_of_week;
            document.getElementById('avgPickups').textContent = data.features.avg_pickups + ' pickups';
            document.getElementById('trendStatus').textContent = data.features.trend.charAt(0).toUpperCase() + data.features.trend.slice(1);
            document.getElementById('zoneInfo').textContent = data.region_name;

            // Update ALL map markers with demand-based colors
            updateMapWithDemand(data.all_regions, data.region_id);

            // console.log(data.recommendations);
            drawRecommendationLine(data);

            // Show Top 5 High Demand Zones
            const topZonesList = document.getElementById('topZonesList');
            topZonesList.innerHTML = '';
            data.top_zones.forEach((zone, index) => {
                const div = document.createElement('div');
                div.style.cssText = `display: flex; justify-content: space-between; align-items: center; padding: 8px; border-radius: 4px; border: 1px solid ${zone.color}; background: ${zone.is_selected ? zone.color + '20' : 'white'}; cursor: pointer;`;
                div.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px; font-weight: 700; color: #9ca3af;">#${index + 1}</span>
                        <div>
                            <p style="font-size: 12px; font-weight: 600; color: #111827;">${zone.name}</p>
                            <p style="font-size: 10px; color: #6b7280;">Zone ${zone.region_id}</p>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <p style="font-size: 18px; font-weight: 700; color: ${zone.color};">${zone.predicted_demand}</p>
                        <p style="font-size: 10px; color: #6b7280;">pickups</p>
                    </div>
                `;
                div.onclick = () => {
                    selectZone(zone.region_id);
                    makePrediction();
                };
                topZonesList.appendChild(div);
            });

            // Show Recommendations for Driver
            const recommendationsList = document.getElementById('recommendationsList');
            recommendationsList.innerHTML = '';

            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach((zone, index) => {
                    const currentDemand = data.prediction.predicted_demand;
                    const diffDemand = zone.predicted_demand - currentDemand;
                    

                    const div = document.createElement('div');
                    div.style.cssText = 'padding: 8px; background: linear-gradient(to right, #f0fdf4, #dbeafe); border-radius: 4px; border: 1px solid #86efac; cursor: pointer;';
                    div.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <p style="font-size: 12px; font-weight: 600; color: #111827;">${zone.name}</p>
                            <span style="
                                background:#10b981;
                                color:white;
                                font-size:10px;
                                font-weight:700;
                                padding:2px 6px;
                                border-radius:2px;">
                                ${zone.badge} • ${zone.recommendation_score.toFixed(1)}
                            </span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px;">
                            <span style="color: #6b7280;">Expected:</span>
                            <span style="font-size: 16px; font-weight: 700; color: ${zone.color};">${zone.predicted_demand}</span>
                        </div>
                        <p style="font-size: 10px; color: #6b7280; margin-top: 4px;">
                            ${zone.distance_km < 1
                                                    ? `<i class="fas fa-map-marker-alt" style="font-size: 8px;"></i> ${Math.round(zone.distance_km * 1000)} m`
                                                    : `<i class="fas fa-map-marker-alt" style="font-size: 8px;"></i> ${zone.distance_km.toFixed(1)} km`
                                                }
                            &nbsp;&nbsp;•&nbsp;&nbsp;
                            <i class="fas fa-clock" style="font-size: 8px;"></i> ${zone.eta_minutes} min
                            &nbsp;&nbsp;•&nbsp;&nbsp;
                            ${diffDemand} more pickups
                        </p>    
                    `;
                    div.onclick = () => {
                        selectZone(zone.region_id);
                        makePrediction();
                    };
                    recommendationsList.appendChild(div);
                });
            } else {
                recommendationsList.innerHTML = '<p style="color: #6b7280; font-size: 12px;">No nearby zones with better demand found.</p>';
            }

            // Recommendation
            let recText = '';
            if (data.prediction.predicted_demand > 80) {
                recText = '🔥 High demand expected! Excellent time for drivers to be in this area.';
            } else if (data.prediction.predicted_demand > 50) {
                recText = '✅ Moderate demand. Good opportunity for pickups in this zone.';
            } else {
                recText = '⚠️ Low demand. Check the "Where to Move Next" section for better zones.';
            }
            document.getElementById('recommendationText').textContent = recText;

            // Update peak hours
            document.getElementById('peakTime').textContent = data.peak_hours.time;
            document.getElementById('peakDemand').textContent = data.peak_hours.demand;

            // Update chart
            updateTrendChart(data.features.lag_values, data.time_labels);
        }

        // Update map markers with demand-based colors (show scatter after prediction)
        async function updateMapWithDemand(allRegions, selectedRegionId) {
            // Clear existing markers
            markers.forEach(marker => map.removeLayer(marker));
            markers = [];

            // Load and show scatter points
            await showScatterPoints();

            // Add selected zone marker
            const selectedRegion = allRegions.find(r => r.region_id === selectedRegionId);
            if (selectedRegion) {
                // Zoom to selected region
                // map.flyTo([selectedRegion.lat, selectedRegion.lon], 13, {
                //     duration: 1.5
                // });

                const marker = L.circleMarker([selectedRegion.lat, selectedRegion.lon], {
                    radius: 6,
                    fillColor: selectedRegion.color,
                    color: '#000',
                    weight: 3,
                    opacity: 1,
                    fillOpacity: 1
                }).addTo(map);

                marker.bindPopup(`
                    <div class="text-center">
                        <b>${selectedRegion.name}</b><br>
                        <span style="color: ${selectedRegion.color}; font-size: 20px; font-weight: bold;">${selectedRegion.predicted_demand}</span> pickups<br>
                        <span class="text-xs bg-purple-100 px-2 py-1 rounded">Selected Zone</span>
                    </div>
                `);
                markers.push(marker);
            }
        }
        async function drawRecommendationLine(data) {

            if (!data.recommendations || data.recommendations.length === 0)
                return;

            if (routeLine)
                map.removeLayer(routeLine);

            const current = data.all_regions.find(
                r => r.region_id === data.region_id
            );

            const best = data.recommendations[0];

            // LOOK UP THE DESTINATION AGAIN FROM all_regions
            const destination = data.all_regions.find(
                r => r.region_id === best.region_id
            );

            console.log("Current:", current);
            console.log("Best (Recommendation):", best);
            console.log("Destination (all_regions):", destination);

            const road = await getRoadRoute(
                current.lat,
                current.lon,
                destination.lat,
                destination.lon
            );
            console.log("Road:", road);
            const roadPoints = road.map(p => [p[1], p[0]]);

            routeLine = L.polyline(roadPoints, {
                color: "#2563eb",
                weight: 6,
                opacity: 0.9
            }).addTo(map);
            routeLine.bringToFront();

            // Jump instantly (NO animation)
            map.fitBounds(routeLine.getBounds(), {
                padding: [50, 50],
                animate: false
            });

            animateTaxi(roadPoints);

            setTimeout(() => {

                taxiMarker.bindPopup(`
                    <div style="
                        width:120px;
                        font-size:10px;
                        line-height:1.2;
                    ">
                        <div style="font-size:9px;font-weight:600">
                            ${best.name}
                        </div>
                        📍 ${best.distance_km.toFixed(1)} km<br>
                        ⏱ ${best.eta_minutes} min
                    </div>
                `, {
                    closeButton: false,
                    autoClose: false,
                    className: "routePopup",
                    offset: [0, -40]
                });

                taxiMarker.openPopup();

            }, 10);

            }
                    
            
            function animateTaxi(points) {


                if (animationTimer) {
                    clearInterval(animationTimer);
                }

                if (taxiMarker) {
                    map.removeLayer(taxiMarker);
                }

                taxiMarker = L.marker(points[0], {
                    icon: taxiIcon
                }).addTo(map);
                taxiMarker.setZIndexOffset(1000);

                // Force browser to draw marker immediately
                taxiMarker.setLatLng(points[0]);

                let i = 0;

                animationTimer = setInterval(() => {

                    const stopPoint = Math.floor(points.length * 0.20);

                    if (i >= stopPoint) {

                        clearInterval(animationTimer);

                        document.getElementById("recommendationText").innerHTML =
                            "🚗 <b>Vehicle En Route</b><br>" +
                            "Continuing towards the recommended demand zone.";

                        return;
                    }

                    i++;

                    const nextIndex = Math.min(i + 1, points.length - 1);

                    taxiMarker.setLatLng(points[nextIndex]);
                    routeLine.setLatLngs(points.slice(nextIndex));

                    map.panTo(points[nextIndex], {
                        animate: true,
                        duration: 0.6
                    });

                    // // Shrink remaining route
                    // remainingRoute.setLatLngs(points.slice(nextIndex));

                    const current = points[nextIndex];
                    const next = points[nextIndex + 1];

                    if (next) {

                        const angle =
                            Math.atan2(
                                next[1] - current[1],
                                next[0] - current[0]
                            ) * 180 / Math.PI;

                        document
                            .getElementById("taxi-car")
                            .style.transform =
                            `rotate(${angle + 180}deg)`;
                    }

                }, 80);

                }

            async function getRoadRoute(startLat, startLon, endLat, endLon) {

                    const url =
                        `https://router.project-osrm.org/route/v1/driving/${startLon},${startLat};${endLon},${endLat}?overview=full&geometries=geojson`;

                    const response = await fetch(url);

                    const data = await response.json();

                    return data.routes[0].geometry.coordinates;
                }

        // Update trend chart
        function updateTrendChart(lagValues, labels) {
            const ctx = document.getElementById('trendChart').getContext('2d');

            if (trendChart) {
                trendChart.destroy();
            }

            trendChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Demand (pickups)',
                        data: lagValues.reverse(),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            titleFont: { size: 14, weight: 'bold' },
                            bodyFont: { size: 13 }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'Pickups', font: { weight: 'bold' } }
                        },
                        x: {
                            title: { display: true, text: 'Time', font: { weight: 'bold' } }
                        }
                    }
                }
            });
        }
        
        // View all zones
        async function viewAllZones() {
            const date = document.getElementById('datePicker').value;
            const time = document.getElementById('timePicker').value;

            if (!date || !time) {
                alert('Please select date and time first');
                return;
            }

            try {
                const response = await fetch('/predict_all_regions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ date, time })
                });
                const data = await response.json();

                if (data.success) {
                    displayAllZones(data.regions);
                } else {
                    alert(data.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error loading all zones');
            }
        }

        // Display all zones
        function displayAllZones(regions) {
            document.getElementById('resultsSection').classList.add('hidden');
            document.getElementById('allZonesSection').classList.remove('hidden');

            const grid = document.getElementById('allZonesGrid');
            grid.innerHTML = '';

            regions.sort((a, b) => b.predicted_demand - a.predicted_demand);

            regions.forEach((region, index) => {
                const card = document.createElement('div');
                card.className = 'bg-white border-2 rounded-lg p-4 hover:shadow-lg transition cursor-pointer';
                card.style.borderColor = region.color;
                card.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <span class="font-semibold text-xs text-gray-600">#${index + 1}</span>
                        <span class="text-2xl font-bold" style="color: ${region.color}">${region.predicted_demand}</span>
                    </div>
                    <h4 class="font-semibold text-sm text-gray-900 mb-1">${region.name}</h4>
                    <p class="text-xs text-gray-600">Actual: ${region.actual_demand}</p>
                `;
                card.onclick = () => {
                    selectZone(region.region_id);
                    document.getElementById('allZonesSection').classList.add('hidden');
                    makePrediction();
                };
                grid.appendChild(card);
            });

            document.getElementById('allZonesSection').scrollIntoView({ behavior: 'smooth' });
        }

        // Event listeners
        document.addEventListener('DOMContentLoaded', () => {
            initMap();
            setTimeout(() => {
                if (map) map.invalidateSize();
            }, 100);
            loadRegions();
            loadAvailableTimes();

            document.getElementById('zoneSelect').addEventListener('change', (e) => {
                if (e.target.value) selectZone(e.target.value);
                checkPredictButton();
            });

            document.getElementById('datePicker').addEventListener('change', () => {
                loadAvailableTimes();
                checkPredictButton();
            });

            document.getElementById('timePicker').addEventListener('change', checkPredictButton);
            document.getElementById('predictBtn').addEventListener('click', makePrediction);
            document.getElementById('viewAllBtn').addEventListener('click', viewAllZones);
            document.getElementById('closeAllZonesBtn').addEventListener('click', () => {
                document.getElementById('allZonesSection').classList.add('hidden');
            });
        });