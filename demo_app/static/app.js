const searchBox = document.getElementById("searchBox");
const statusFilter = document.getElementById("statusFilter");
const table = document.getElementById("resultsTable");

function applyFilters() {
  if (!table) return;

  const query = (searchBox?.value || "").trim().toLowerCase();
  const status = statusFilter?.value || "all";
  const rows = table.querySelectorAll("tbody tr");

  rows.forEach((row) => {
    const searchable = (row.dataset.search || "").toLowerCase();
    const rowStatus = row.dataset.status || "";
    const matchesQuery = !query || searchable.includes(query);
    const matchesStatus = status === "all" || rowStatus === status;
    row.style.display = matchesQuery && matchesStatus ? "" : "none";
  });
}

searchBox?.addEventListener("input", applyFilters);
statusFilter?.addEventListener("change", applyFilters);
