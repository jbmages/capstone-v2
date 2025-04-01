// Load the dataset
d3.csv("../data/data_with_clusters.csv").then(function(data) {
    console.log("Raw Data Loaded:", data);

    if (!data || data.length === 0) {
        console.error("Error: No data loaded. Check your CSV path or format.");
        return;
    }

    // Extract first 50 columns + last 2 (cluster assignments)
    let allColumns = Object.keys(data[0]);
    console.log("All Columns Detected:", allColumns);

    // Ensure "KMeans_Cluster" and "GMM_Cluster" exist
    let clusterColumns = ["KMeans_Cluster", "GMM_Cluster"].filter(col => allColumns.includes(col));

    if (clusterColumns.length < 2) {
        console.error("Error: Cluster columns not found in dataset.");
        return;
    }

    let selectedColumns = allColumns.slice(0, 50).concat(clusterColumns);
    console.log("Columns Selected:", selectedColumns);

    // Select the table container
    let table = d3.select("#data-table").append("table").attr("class", "styled-table");

    // Append the header row
    let thead = table.append("thead");
    let tbody = table.append("tbody");

    thead.append("tr")
        .selectAll("th")
        .data(selectedColumns)
        .enter()
        .append("th")
        .text(d => d);

    // Populate the table with rows
    data.forEach(row => {
        let tr = tbody.append("tr");
        selectedColumns.forEach(col => {
            tr.append("td").text(row[col]);
        });
    });

    console.log("Table rendered.");
}).catch(error => {
    console.error("Error loading CSV file:", error);
});
