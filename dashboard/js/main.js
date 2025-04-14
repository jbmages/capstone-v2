// loadClusterData(function(data) {
    

//     const numericCols = Object.keys(data[0]).slice(0, 50);
//     const clusterOptions = [
//         { label: "KMeans Cluster", value: "KMeans_Cluster" },
//         { label: "GMM Cluster", value: "GMM_Cluster" }
//     ];

//     setupDropdowns(data, numericCols, clusterOptions, updateChart);

//     const svg = d3.select("#chart");
//     const x = d3.scaleLinear().range([40, 580]);
//     const y = d3.scaleLinear().range([360, 40]);

//     function updateChart() {
//         const column = d3.select("#columnDropdown").property("value");
//         const clusterMethod = d3.select("#clusterDropdown").property("value");

//         const values = data.map(d => ({
//             value: +d[column],
//             cluster: d[clusterMethod]
//         })).filter(d => !isNaN(d.value));

//         const uniqueClusters = Array.from(new Set(values.map(d => d.cluster))).sort((a, b) => a - b);
//         const color = d3.scaleOrdinal()
//             .domain(uniqueClusters)
//             .range(d3.schemeTableau10.concat(d3.schemeSet3).slice(0, uniqueClusters.length));

//         x.domain(d3.extent(values.map(d => d.value)));

//         drawHistogram(svg, data, column, clusterMethod, color, x, y);
//         drawLegend(uniqueClusters, color);
//     }

    

//     updateChart(); // Initial chart
// });

loadClusterData(function(data) {
    const numericCols = Object.keys(data[0]).slice(0, 50);

    const clusterOptions = [
        { label: "KMeans Cluster", value: "KMeans_Cluster" },
        { label: "GMM Cluster", value: "GMM_Cluster" }
    ];

    const svg = d3.select("#chart");
    const x = d3.scaleLinear().range([40, 580]);
    const y = d3.scaleLinear().range([360, 40]);

    function updateChart() {
        const column = d3.select("#columnDropdown").property("value");
        const clusterMethod = d3.select("#clusterDropdown").property("value");

        console.log("Updating chart with:", column, clusterMethod);

        const values = data.map(d => ({
            value: +d[column],
            cluster: d[clusterMethod]
        })).filter(d => !isNaN(d.value) && d.cluster !== undefined);

        const uniqueClusters = Array.from(new Set(values.map(d => d.cluster))).sort((a, b) => a - b);
        const color = d3.scaleOrdinal()
            .domain(uniqueClusters)
            .range(d3.schemeTableau10.concat(d3.schemeSet3).slice(0, uniqueClusters.length));

        x.domain(d3.extent(values.map(d => d.value)));

        drawHistogram(svg, data, column, clusterMethod, color, x, y);
        drawLegend(uniqueClusters, color);
        console.log("Updating chart with", column, "and", clusterMethod);

    }

    setupDropdowns(data, numericCols, clusterOptions, updateChart);

    updateChart();
});
