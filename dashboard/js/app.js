d3.json("dash-data/cluster_data.json").then(function(loadedData) {
    const data = loadedData;
    const traitGroupDescriptions = {
    "EXT": "Extraversion reflects how outgoing, sociable, and energetic a person is. High scorers tend to be assertive and thrive in social environments.",
    "EST": "Emotional Stability relates to emotional volatility. Higher scores indicate greater sensitivity to stress and mood swings.",
    "AGR": "Agreeableness measures warmth, kindness, and cooperation. High scorers are generally empathetic, generous, and supportive.",
    "CSN": "Conscientiousness describes how organized, reliable, and goal-driven a person is. High scorers are often focused, hardworking, and disciplined.",
    "OPN": "Openness to Experience reflects imagination, curiosity, and a preference for variety and novelty."
};
    console.log("Data Loaded:", data);
    if (!data || data.length === 0) {
        console.error("Error: No data loaded.");
        return;
    }

    let numericCols = Object.keys(data[0]).slice(0, 50);
    const clusterOptions = ['KMeans_Cluster', 'GMM_Cluster'];
    const methodDropdown = d3.select("#clusterDropdown");

    methodDropdown.selectAll("option")
        .data(clusterOptions)
        .enter()
        .append("option")
        .text(d => d.replace("_", " "))  
        .attr("value", d => d);        

    // select column selection dropdown
    const columnDropdown = d3.select("#columnDropdown");


    columnDropdown.selectAll("option")
        .data(numericCols)
        .enter()
        .append("option")
        .text(d => d)
        .attr("value", d => d);

    let svg = d3.select("#chart")
    .attr("width", 600)
    .attr("height", 450);

    let legend = d3.select("#legend");


    function updateChart(col, clusterMethod) {

        const values = data.map(d => ({
            value: +d[col],
            cluster: d[clusterMethod]
        })).filter(d => !isNaN(d.value) && d.cluster !== undefined);


        svg.selectAll("*").remove();
        legend.selectAll("*").remove();

        // retrieve unique clusters
        let uniqueClusters = Array.from(new Set(values.map(d => d.cluster))).sort((a, b) => a - b);

        // color scale
        let color = d3.scaleOrdinal()
            .domain(uniqueClusters)
            .range(d3.schemeTableau10.concat(d3.schemeSet3).slice(0, uniqueClusters.length)); // expand for more clusters

        // X scale
        let x = d3.scaleBand()
            .domain([1, 2, 3, 4, 5])
            .range([60, 340])  
            .padding(0.2);

        // bins for each cluster separately
        let binGenerator = d3.bin()
            .domain([0.5, 5.5])
            .thresholds([1, 2, 3, 4, 5, 6]);

        let clusteredBins = {};
        uniqueClusters.forEach(cluster => {
            let vals = values.filter(d => d.cluster === cluster).map(d => d.value);
            clusteredBins[cluster] = binGenerator(vals);
        });

        // max y value
        let allBinHeights = [];
        Object.values(clusteredBins).forEach(bins => {
            bins.forEach((bin, i) => {
                allBinHeights[i] = (allBinHeights[i] || 0) + bin.length;
            });
        });

        let y = d3.scaleLinear()
            .domain([0, d3.max(allBinHeights)])
            .range([360, 40]);

        // histogram bars
        let binKeys = clusteredBins[uniqueClusters[0]].map((_, i) => i);

        binKeys.forEach(i => {
            let x0 = clusteredBins[uniqueClusters[0]][i].x0;
            let x1 = clusteredBins[uniqueClusters[0]][i].x1;
            let xPos = x(x0) + 1;
            let binWidth = x(x1) - x(x0) - 2;
            let yOffset = 360;

            uniqueClusters.forEach(cluster => {
                let bin = clusteredBins[cluster][i];
                let height = 360 - y(bin.length);
                svg.append("rect")
                    .attr("x", xPos)
                    .attr("y", yOffset - height)
                    .attr("width", binWidth)
                    .attr("height", height)
                    .attr("fill", color(cluster));
                yOffset -= height;
            });
        });

        // X axis
        svg.append("g")
            .attr("transform", "translate(0,360)")
            .call(d3.axisBottom(x).tickValues([1, 2, 3, 4, 5]));

        // X axis label
        svg.append("text")
            .attr("x", 300)
            .attr("y", 440)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text("Survey Response (1 = Disagree, 5 = Agree)");

        // Y axis
        svg.append("g")
            .attr("transform", "translate(40,0)")
            .call(d3.axisLeft(y));

        // Y axis label
        svg.append("text")
            .attr("text-anchor", "middle")
            .attr("transform", "rotate(-90)")
            .attr("x", -200)
            .attr("y", 5)
            .style("font-size", "12px")
            .text("Number of Participants out Subset");

        // title
        svg.append("text")
            .attr("x", 300)
            .attr("y", 20)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(`Distribution of ${col} by ${clusterMethod}`);

        // dynamic trait description
        const prefix = col.slice(0, 3); 
        const description = traitGroupDescriptions[prefix] || "This trait is part of the Big Five personality model.";
        d3.select("#trait-description").text(description);

        legend.selectAll("div")
            .data(uniqueClusters)
            .enter()
            .append("div")
            .style("display", "inline-block")
            .style("margin-right", "10px")
            .html(d => `<span style="display:inline-block;width:12px;height:12px;background-color:${color(d)};margin-right:4px;"></span>Cluster ${+d + 1}`);
    }

    // first chart
    updateChart(numericCols[0], clusterOptions[0]);

    // on dropdown change
    methodDropdown.on("change", function() {
        const col = columnDropdown.property("value");
        const cluster = this.value;
        updateChart(col, cluster);
    });

    columnDropdown.on("change", function() {
        const col = this.value;
        const cluster = methodDropdown.property("value");
        updateChart(col, cluster);
    });

}).catch(function(error) {/// debugging
    console.error("Error loading the JSON file:", error);
});
