// check if data loaded
window.loadClusterData = function(callback) {
    d3.json("dash-data/cluster_data.json").then(function(data) {
        if (!data || data.length === 0) {
            console.error("Error: No data loaded.");
            return;
        }
        callback(data);
    }).catch(function(error) {
        console.error("Error loading JSON file:", error);
    });
}
