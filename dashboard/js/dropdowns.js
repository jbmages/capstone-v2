window.setupDropdowns = function(data, numericCols, clusterOptions, onChange) {
    const clusterDropdown = d3.select("#clusterDropdown");
    const columnDropdown = d3.select("#columnDropdown");

    clusterDropdown.selectAll("option")
        .data(clusterOptions)
        .enter()
        .append("option")
        .text(d => d)
        .attr("value", d => d);

    columnDropdown.selectAll("option")
        .data(numericCols)
        .enter()
        .append("option")
        .text(d => d)
        .attr("value", d => d);

    clusterDropdown.on("change", () => onChange());
    columnDropdown.on("change", () => onChange());
}
