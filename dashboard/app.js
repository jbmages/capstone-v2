// Load the JSON dataset
d3.json("data/cluster_data.json").then(function(data) {
    console.log("Data Loaded:", data);

    // Check if data is loaded
    if (!data || data.length === 0) {
        console.error("Error: No data loaded. Check your JSON path or format.");
        return;
    }

    // Subset the first 50 columns + last 2 (cluster assignments)
    let columns = Object.keys(data[0]).slice(0, 50).concat(["KMeans_Cluster", "GMM_Cluster"]);

    console.log("Columns Extracted:", columns);

    // Select the table container
    let table = d3.select("#data-table").append("table").attr("class", "styled-table");

    // Append the header row
    let thead = table.append("thead");
    let tbody = table.append("tbody");

    thead.append("tr")
        .selectAll("th")
        .data(columns)
        .enter()
        .append("th")
        .text(d => d);

    // Populate the table with rows
    data.forEach(row => {
        let tr = tbody.append("tr");
        columns.forEach(col => {
            tr.append("td").text(row[col]);
        });
    });

    console.log("Table rendered.");
}).catch(function(error) {
    console.error("Error loading the JSON file:", error);
});
