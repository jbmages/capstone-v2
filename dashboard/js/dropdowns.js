// window.setupDropdowns = function(data, numericCols, clusterOptions, onChange) {
//     const clusterDropdown = d3.select("#clusterDropdown");
//     const columnDropdown = d3.select("#columnDropdown");

//     clusterDropdown.selectAll("option")
//         .data(clusterOptions)
//         .enter()
//         .append("option")
//         .text(d => d.label)
//         .attr("value", d => d.value);

//     columnDropdown.selectAll("option")
//         .data(numericCols)
//         .enter()
//         .append("option")
//         .text(d => d)
//         .attr("value", d => d);

//     clusterDropdown.on("change", () => onChange());
//     columnDropdown.on("change", () => onChange());
// }
console.log("dropdowns.js loaded!");
window.setupDropdowns = function(data, numericCols, clusterOptions, onChange) {
    console.log("Setting up dropdowns...");

    const clusterDropdown = d3.select("#clusterDropdown");
    const columnDropdown = d3.select("#columnDropdown");

    clusterDropdown.selectAll("option")
        .data(clusterOptions)
        .enter()
        .append("option")
        .text(d => d.label)
        .attr("value", d => d.value);

    columnDropdown.selectAll("option")
        .data(numericCols)
        .enter()
        .append("option")
        .text(d => d)
        .attr("value", d => d);

    // chart updates when dropdown changes
   clusterDropdown.on("change", () => {
        console.log("Cluster dropdown changed");
        onChange();
    });

    columnDropdown.on("change", () => {
        console.log("Column dropdown changed");
        onChange();
    });
}
