# visjs_component,py
import streamlit.components.v1 as components
import json

def visjs_graph(nodes, edges, options=None):
    """
    Renders an interactive graph using Vis.js.

    Args:
        nodes: A list of node objects.
        edges: A list of edge objects.
        options: A dictionary of options for the Vis.js network.
    """
    # Convert data to JSON strings
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    options_json = json.dumps(options) if options else "{}"

    # HTML template for the component
    html_code = f"""
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style type="text/css">
            #mynetwork {{
                width: 100%;
                height: 500px;
                border: 1px solid lightgray;
            }}
            .vis-node .fa {{
                font-size: 16px;
                margin-right: 5px;
            }}
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            // create an array with nodes
            var nodes = new vis.DataSet({nodes_json});

            // create an array with edges
            var edges = new vis.DataSet({edges_json});

            // create a network
            var container = document.getElementById('mynetwork');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            var options = {options_json};

            // Add node click event
            options.interaction = {{
                hover: true
            }};
            options.nodes = {{
                shape: 'box'
            }};

            // Add context menu for canceling tasks
            options.manipulation = {{
                enabled: false,
                addNode: false,
                addEdge: false,
                editEdge: false,
                deleteNode: false,
                deleteEdge: false,
                controlNodeStyle: {{
                    // all node options are valid.
                }}
            }};

            var network = new vis.Network(container, data, options);

            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var node = nodes.get(nodeId);
                    // Call Python function to display task details
                    window.parent.Streamlit.setComponentValue("showTaskDetails", node.id);
                }}
            }});

            // Add context menu for task actions
            network.on("oncontext", function (params) {{
                params.event.preventDefault();
                var nodeId = network.getNodeAt(params.pointer.DOM);
                if (nodeId) {{
                    var node = nodes.get(nodeId);
                    // Show context menu
                    var menu = document.createElement('div');
                    menu.style.position = 'absolute';
                    menu.style.left = params.pointer.DOM.x + 'px';
                    menu.style.top = params.pointer.DOM.y + 'px';
                    menu.innerHTML = `
                        <button onclick="pauseTask('${{node.id}}')">Pause Task</button>
                        <button onclick="resumeTask('${{node.id}}')">Resume Task</button>
                        <button onclick="cancelTask('${{node.id}}')">Cancel Task</button>
                        <button onclick="viewResults('${{node.id}}')">View Results</button>
                    `;
                    document.body.appendChild(menu);

                    // Hide menu on click outside
                    document.addEventListener('click', function(event) {{
                        if (!menu.contains(event.target)) {{
                            document.body.removeChild(menu);
                        }}
                    }});
                }}
            }});

            function pauseTask(taskId) {{
                // Send pause command to Python
                window.parent.Streamlit.setComponentValue("pauseTask", taskId);
            }}

            function resumeTask(taskId) {{
                // Send resume command to Python
                window.parent.Streamlit.setComponentValue("resumeTask", taskId);
            }}

            function cancelTask(taskId) {{
                // Send cancel command to Python
                window.parent.Streamlit.setComponentValue("cancelTask", taskId);
            }}

            function viewResults(taskId) {{
                // Send view results command to Python
                window.parent.Streamlit.setComponentValue("viewResults", taskId);
            }}
        </script>
    </body>
    </html>
    """

    # Render the component using components.html
    components.html(html_code, height=500)
