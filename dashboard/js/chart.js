const x = d3.scaleBand()
    .domain([1, 2, 3, 4, 5])
    .range([60, 700])
    .padding(0.2);

window.drawHistogram = function(svg, data, column, clusterMethod, color, x, y) {
    svg.selectAll("*").remove();

    const values = data.map(d => ({
        value: +d[column],
        cluster: d[clusterMethod]
    })).filter(d => !isNaN(d.value));

    const uniqueClusters = Array.from(new Set(values.map(d => d.cluster))).sort((a, b) => a - b);

    // match Likert-scale values (1–5)
    const binGenerator = d3.bin()
        .domain([0.5, 5.5])
        .thresholds([1, 2, 3, 4, 5, 6]);

    let clusteredBins = {};
    uniqueClusters.forEach(cluster => {
        let vals = values.filter(d => d.cluster === cluster).map(d => d.value);
        clusteredBins[cluster] = binGenerator(vals);
    });

    let allBinHeights = Array(5).fill(0);
    uniqueClusters.forEach(cluster => {
        clusteredBins[cluster].forEach((bin, i) => {
            allBinHeights[i] += bin.length;
        });
    });

    y.domain([0, d3.max(allBinHeights)]);
    x.domain([1, 2, 3, 4, 5]);
// console.log("x.bandwidth() =", x.bandwidth()); // debugging

// spacing fixes
    for (let i = 0; i < 5; i++) {
        const xMid = i + 1;
        const xPos = x(xMid);
        const binWidth = x.bandwidth();
        let yOffset = 360;

        uniqueClusters.forEach(cluster => {
            const bin = clusteredBins[cluster][i];
            const height = 360 - y(bin.length);
            svg.append("rect")
                .attr("x", xPos)
                .attr("y", yOffset - height)
                .attr("width", binWidth)
                .attr("height", height)
                .attr("fill", d3.color(color(cluster)).darker(0.3))
                .style("stroke", "#fff")
                .style("stroke-width", 0.5)
                .style("filter", "drop-shadow(1px 1px 2px rgba(0,0,0,0.15))");
            yOffset -= height;
        });
    }

    svg.append("g")
        .attr("transform", "translate(0,360)")
        .call(d3.axisBottom(x).tickValues([1, 2, 3, 4, 5]));

    svg.append("g")
        .attr("transform", "translate(40,0)")
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("x", 300)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .text(`Distribution of ${column} by ${clusterMethod}`);
};
