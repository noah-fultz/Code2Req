
function openSidePanel(title, blocks) {
    let content = "";
    for (const block of blocks) {
        const label = block.title.toLowerCase();
        if (label === "code") {
        content += `
            <p><strong>Code:</strong></p>
            <pre class="bg-light p-2 border rounded" style="white-space: pre-wrap;"><code>${block.content}</code></pre>
        `;
        } else if (label== "id"){
            content += `<p><strong>ID:</strong> <span id="current_id">${block.content}</span></p>`;
        }
        else if (label== "type"){
            content += `<p><strong>Type:</strong>${block.content}</span></p>`;
            content += "<button class='btn btn-outline-secondary' type='button' onClick='getChildren()' id='getChildrenButton'>View in Network</button><br><br>";
        }
        else {
        content += `
            <p><strong>${block.title}:</strong></p>
            <div class="mb-2">${block.content}</div>
        `;
        }
    }
  
    document.getElementById("sidePanelTitle").innerText = title;
    document.getElementById("sidePanelContent").innerHTML = content;
    document.getElementById("sidePanel").style.display = "block";
  }

  function closeSidePanel() {
    document.getElementById("sidePanel").style.display = "none";
  }
  
  function fetchData(id) {
    fetch(`/source-detail/?uri=${encodeURIComponent(id)}`)
      .then(response => response.json())
      .then(data => {
        openSidePanel(data.title || "Detail", data.blocks || []);
      })
      .catch(err => {
        openSidePanel("Error", [{ title: "Error", content: "Could not load source detail." }]);
        console.error(err);
      });
  }

  function getChildren() {
    const query = document.getElementById("current_id").innerHTML;
    if (!query) return;
    window.location.href = `/network/?query=${encodeURIComponent(query)}`;
  }
  