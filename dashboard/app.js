d3.json("dash-data/cluster_data.json").then(function(data) {
    console.log("Data Loaded:", data);
    if (!data || data.length === 0) {
        console.error("Error: No data loaded.");
        return;
    }

    // Extract the first 50 columns for dropdown
    let numericCols = Object.keys(data[0]).slice(0, 50);

    // Create dropdown
    let dropdown = d3.select("body")
        .insert("div", "#data-table")
        .attr("id", "dropdown-container")
        .append("select")
        .attr("id", "columnDropdown");

    dropdown.selectAll("option")
        .data(numericCols)
        .enter()
        .append("option")
        .text(d => d)
        .attr("value", d => d);

    // Add SVG for the chart
    d3.select("body")
        .insert("div", "#data-table")
        .attr("id", "chart-container")
        .append("svg")
        .attr("width", 600)
        .attr("height", 400);

    function updateChart(col) {
        let values = data.map(d => +d[col]).filter(v => !isNaN(v));
        let svg = d3.select("svg");
        svg.selectAll("*").remove();

        // X scale for histogram bins
        let x = d3.scaleLinear()
            .domain(d3.extent(values))
            .range([40, 580]);

        // Create bins
        let bins = d3.bin()
            .domain(x.domain())
            .thresholds(20)(values);

        // Y scale
        let y = d3.scaleLinear()
            .domain([0, d3.max(bins, d => d.length)])
            .range([360, 40]);

        // Draw histogram bars
        svg.selectAll("rect")
            .data(bins)
            .enter()
            .append("rect")
            .attr("x", d => x(d.x0))
            .attr("y", d => y(d.length))
            .attr("width", d => x(d.x1) - x(d.x0) - 1)
            .attr("height", d => 360 - y(d.length))
            .attr("fill", "#69b3a2");

        // X axis
        svg.append("g")
            .attr("transform", "translate(0,360)")
            .call(d3.axisBottom(x));

        // Y axis
        svg.append("g")
            .attr("transform", "translate(40,0)")
            .call(d3.axisLeft(y));

        // Fit normal distribution
        let mean = d3.mean(values);
        let stdDev = d3.deviation(values);

        let normalLine = d3.line()
            .x(d => x(d))
            .y(d => {
                let z = (d - mean) / stdDev;
                let p = (1 / (Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z);
                return y(p * values.length * (x.domain()[1] - x.domain()[0]) / 20); // scale to match histogram
            });

        let lineData = d3.range(x.domain()[0], x.domain()[1], (x.domain()[1] - x.domain()[0]) / 100);

        svg.append("path")
            .datum(lineData)
            .attr("fill", "none")
            .attr("stroke", "red")
            .attr("stroke-width", 2)
            .attr("d", normalLine);

        // Chart title
        svg.append("text")
            .attr("x", 300)
            .attr("y", 20)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(`Distribution of ${col}`);
    }

    // Initial chart
    updateChart(numericCols[0]);

    // Change on dropdown
    dropdown.on("change", function() {
        let selectedCol = d3.select(this).property("value");
        updateChart(selectedCol);
    });

    // Optional: retain your data table render
    // ... (your original table-rendering code here if needed) ...

}).catch(function(error) {
    console.error("Error loading the JSON file:", error);
});
