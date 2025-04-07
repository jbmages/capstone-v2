window.drawHistogram = function(svg, data, column, clusterMethod, color, x, y) {
    svg.selectAll("*").remove();

    const values = data.map(d => ({
        value: +d[column],
        cluster: d[clusterMethod]
    })).filter(d => !isNaN(d.value));

    const uniqueClusters = Array.from(new Set(values.map(d => d.cluster))).sort((a, b) => a - b);
    const binGenerator = d3.bin().domain(x.domain()).thresholds(20);

    let clusteredBins = {};
    uniqueClusters.forEach(cluster => {
        let vals = values.filter(d => d.cluster === cluster).map(d => d.value);
        clusteredBins[cluster] = binGenerator(vals);
    });

    let allBinHeights = [];
    Object.values(clusteredBins).forEach(bins => {
        bins.forEach((bin, i) => {
            allBinHeights[i] = (allBinHeights[i] || 0) + bin.length;
        });
    });

    y.domain([0, d3.max(allBinHeights)]);

    const binKeys = clusteredBins[uniqueClusters[0]].map((_, i) => i);

    binKeys.forEach(i => {
        let x0 = clusteredBins[uniqueClusters[0]][i].x0;
        let x1 = clusteredBins[uniqueClusters[0]][i].x1;
        let xPos = x(x0);
        let binWidth = x(x1) - x(x0) - 1;
        let yOffset = 360;

        uniqueClusters.forEach(cluster => {
            let bin = clusteredBins[cluster][i];
            if (bin.length > 0) {
                let height = 360 - y(bin.length);
                svg.append("rect")
                    .attr("x", xPos)
                    .attr("y", yOffset - height)
                    .attr("width", binWidth)
                    .attr("height", height)
                    .attr("fill", color(cluster));
                yOffset -= height;
            }
        });
    });

    svg.append("g")
        .attr("transform", "translate(0,360)")
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", "translate(40,0)")
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("x", 300)
        .attr("y", 20)
        .attr("text-anchor", "middle")
        .style("font-size", "16px")
        .text(`Distribution of ${column} by ${clusterMethod}`);
}
