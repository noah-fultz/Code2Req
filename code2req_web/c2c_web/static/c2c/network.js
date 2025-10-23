let cy;
const typeColors = [
  '#FFCA28',  // Soft Yellow-Orange (Bright Accent, no purple!)
  '#26C6DA', // Teal (Cool Light Blue)
  '#EA4335', // Red (Google Red)
  '#FBBC05', // Yellow (Google Yellow)
  '#34A853', // Green (Google Green)
  '#FF7043', // Orange (Vibrant)
  '#4285F4', // Blue (Google Blue)
  '#4CAF50', // Medium Green (Strong Fresh)
  '#29B6F6', // Sky Blue (Bright Fresh)
  '#00ACC1', // Cyan (Fresh Blue-Cyan)
];
  
  

const styleRules = typeColors.map((color, index) => {
    return {
      selector: `node[styleClass = "type:${index}"]`,
      style: { 'background-color': color }
    };
  });

function renderGraph(data) {
  if (cy) {
    cy.destroy();
  }

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: data,
    style: [
        {
            selector: 'node',
            style: {
              'label': 'data(label)',
              'text-valign': 'center',
              'color': '#000',
              'font-size': '4px',
              'text-outline-color': '#fff',
              'text-outline-width': 1
            }
          },
          {
            selector: 'edge',
            style: {
              'label': 'data(label)', 
              'curve-style': 'bezier', 
              'target-arrow-shape': 'triangle',  
              'target-arrow-color': '#999',
              'line-color': '#ccc',
              'font-size': '4px',
              'text-rotation': 'autorotate',    
              'text-margin-y': -4,
            }
          },
          ...styleRules
    ],
    layout: {
        name: 'cose',
        animate: true,
        animationDuration: 5000,
        nodeRepulsion: 8000,
        idealEdgeLength: 100,
        gravity: 0.8,
        numIter: 1000
      }
  });

  cy.on('tap', 'node', function(evt) {
    const node = evt.target;
    const label = node.data('label');
    const type = node.data('type');
    const id = node.data('id');
    openSidePanel("Loading source...", []);
    fetchData(id);
  });

cy.on('dbltap', 'node', function(evt) {
  const node = evt.target;
  const id = node.data('id');
  if (!id) return;

  fetch(`/network/expand/?query=${encodeURIComponent(id)}`)
    .then(res => res.json())
    .then(data => {
      expandGraph(data);
    });
});


  
}

// Initial render from template
renderGraph(initialData);

function expandGraph(data) {
  const existingIds = new Set(cy.nodes().map(n => n.id()));

  const newNodes = data.nodes.filter(n => !existingIds.has(n.data.id));
  const newEdges = data.edges.filter(e => {
    return !cy.getElementById(e.data.id).nonempty(); // prevent duplicates
  });

  cy.add([...newNodes, ...newEdges]);

  // Optionally position new nodes relative to their source node
  for (const node of newNodes) {
    const relatedEdges = newEdges.filter(e =>
      e.data.source === node.data.id || e.data.target === node.data.id
    );

    if (relatedEdges.length > 0) {
      const refEdge = relatedEdges[0];
      const refNodeId = refEdge.data.source === node.data.id
        ? refEdge.data.target
        : refEdge.data.source;

      const refNode = cy.getElementById(refNodeId);
      if (refNode) {
        cy.getElementById(node.data.id).position({
          x: refNode.position('x') + 50 + Math.random() * 50,
          y: refNode.position('y') + 50 + Math.random() * 50
        });
      }
    }
  }

  // Run layout ONLY on new nodes if needed
  cy.layout({
    name: 'preset', // keep positions
    fit: false
  }).run();
}


document.addEventListener("DOMContentLoaded", function () {
  const params = new URLSearchParams(window.location.search);
  const query = params.get("query");
  console.log(query)

  if (query && query.startsWith("QUERY:")){
    // Example 1: prefill a form field
    const inputField = document.getElementById("nodeSearch");
    if (inputField) {
      const cleanedValue = decodeURIComponent(query.replace(/^QUERY:/, ""));
      inputField.value = decodeURIComponent(cleanedValue);
    }
  } else if (query && query.startsWith("FILTER:")){
      const inputField = document.getElementById("nodeFilter");
      if (inputField) {
        const cleanedValue = decodeURIComponent(query.replace(/^FILTER:/, ""));
        inputField.value = decodeURIComponent(cleanedValue);
      }
  }
  else if (query){
    openSidePanel("Loading source...", []);
    fetchData(decodeURIComponent(query));
  }
});

// Search handler
document.getElementById("searchButton").addEventListener("click", () => {
    const query = document.getElementById("nodeSearch").value.trim();
    if (!query) return;
    window.location.href = `/network/?query=QUERY:${encodeURIComponent(query)}`;
  });
  

// Dropdown handler
document.addEventListener('DOMContentLoaded', function () {
  const nodeFilter = document.getElementById('nodeFilter');

  if (nodeFilter) {
    nodeFilter.addEventListener('change', function () {
      const selectedType = this.value;
      window.location.href = `/network/?query=FILTER:${selectedType}`;
    });
  }
});
