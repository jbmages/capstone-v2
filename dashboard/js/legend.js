window.drawLegend = function(clusters, color) {
    const legend = d3.select("#legend");
    legend.selectAll("*").remove();

    legend.selectAll("div")
        .data(clusters)
        .enter()
        .append("div")
        .style("display", "inline-block")
        .style("margin-right", "10px")
        .html(d => `<span style="display:inline-block;width:12px;height:12px;background-color:${color(d)};margin-right:4px;"></span>Cluster ${+d + 1}`);
}
