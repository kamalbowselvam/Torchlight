from graphviz import Digraph



def draw_dag(out): 
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'}) #, node_attr={'rankdir': 'TB'})
    for n in out.nodes:
        dot.node(name=str(id(n)), label = "{ %s | %s }" % (n.label, n._data.shape), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
            
        #if n.opsymbol:
        #    dot.node(name=str(id(n)) + n.opsymbol, label=n.opsymbol)
        #    dot.edge(str(id(n)) + n.opsymbol, str(id(n)))

    for n1, n2 in out.edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        dot.render(directory='output')