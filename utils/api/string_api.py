from io import StringIO
from time import sleep
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import requests


def get_ppi_encoder(chosen_genes: list[str], score: str = 'escore', threshold: float = 0.0):
    """Get PPI for chosen genes

    Args:
        chosen_genes (list[str]): list of chosen genes

    Returns:
        ppi (pd.DataFrame): PPI network
    """

    # I think the setting on tcga project dataset and program dataset are overriding the "score" parameter
    
    score = "escore" #delete this later

    gene_encoder = dict(zip(chosen_genes, range(len(chosen_genes))))

    res_text = _get_ppi_network_from_string(chosen_genes)

    ppi = pd.read_csv(StringIO(res_text), sep='\t')
    ppi = ppi[['preferredName_A', 'preferredName_B', score]]
   
    #filter by genes   
    ppi.drop_duplicates(inplace=True)       
    ppi = ppi.reset_index(drop=True)

    # visualize network before filtering
    visualize_ppi(ppi, score=score, threshold=0.0)
    #filter by threshold
    # ppi = ppi[ppi[score] >= threshold]

    # # visualize network after filtering
    # visualize_ppi(ppi, score=score, threshold=threshold)


    # ppi = ppi[ppi[score] >= 0.7]
    # visualize_ppi(ppi, score=score, threshold=0.7)

   
    #ppi[['src', 'dst']] = ppi[['preferredName_A', 'preferredName_B']].map(lambda x: gene_encoder[x])
    ppi[['src', 'dst']] = ppi[['preferredName_A', 'preferredName_B']].applymap(lambda x: gene_encoder[x]) #different versions of pandas
    return ppi


def _get_ppi_network_from_string(gene_list: list[str]):
    """Get PPI network from STRING API.

    Args:
        gene_list (list[str]): list of genes

    Returns:
        res_text (str): response from STRING API in text format
    """
    string_api_url = 'https://string-db.org/api'
    request_url = '/'.join([string_api_url, 'tsv', 'network'])

    # Speed up the process by getting the identifier first.
    if len(gene_list) > 100:
        gene_list = _get_identifier(gene_list, string_api_url).values()

    params = {
        'identifiers': '%0d'.join(gene_list),           # Genes
        'species': 9606,                                # Species NCBI identifier for homo sapiens
        'caller_identity': 'www.idssp.ee.ntu.edu.tw'    # App name
    }

    # Link to the graph.
    # request_url = '/'.join([string_api_url, 'tsv-no-header', 'get_link'])
    # response = requests.post(request_url, data=params)

    response = requests.post(request_url, data=params)
    if response.status_code != 200:
        raise ConnectionError(f'Response from STRING: {response.status_code}')
    return response.text


def _get_identifier(gene_list: list[str], string_api_url: str):
    """Each gene has corresponding identifiers in STRING. Get the best one out of them.

    Args:
        gene_list (list[str]): list of genes
        string_api_url (str): String API

    Returns:
        identifiers (dict[str, str]): corresponding identifiers
    """
    params = {
        'identifiers': '\r'.join(gene_list),            # Genes
        'species': 9606,                                # Species NCBI identifier Homo Sapiens
        'limit': 1,                                     # Return only one (best) identifier per protein
        'caller_identity': 'www.idssp.ee.ntu.edu.tw'    # App name
    }
    request_url = '/'.join([string_api_url, 'tsv-no-header', 'get_string_ids'])
    response = requests.post(request_url, data=params)

    if response.status_code != 200:
        raise ConnectionError(f'Responce from STRING: {response.status_code}')

    identifiers: dict[str, str] = {}
    for line in response.text.strip().split('\n'):
        counter, identifier, species_id, species, gene, meaning = line.split('\t')
        identifiers[gene] = identifier
    if set(identifiers.keys()) != set(gene_list):
        raise ValueError('Cannot get correct identifiers.')
    return identifiers

def get_network_image(gene_list: list[str], min_score: float = 0.7):
    """
    Get image directly from STRINGDB

    """

    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "image"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])
    identifiers = "%0D".join(gene_list)

    ## For each gene call STRING

    

    params = {

        "identifiers" : identifiers, # protein
        "species" : 9606, # species NCBI identifier 
        "edge_score_cutoff": min_score,  # filter by minimum score
        'limit': 1,    # Return only one (best) identifier per protein 
        "network_flavor": "evidence", # show confidence links
        "caller_identity" :'www.idssp.ee.ntu.edu.tw' 

        }

    ## Call STRING       

    response = requests.post(request_url, data=params)
    if response.status_code != 200:
        raise ConnectionError(f'Response from STRING: {response.status_code}')

    ## Save the network to file

    file_name = "%s_network.png" % str(min_score)
    print(f"Saving interaction network to {file_name}")

    with open(file_name, 'wb') as fh:
        fh.write(response.content)
    
    sleep(1)




#visualization with node degree
# def visualize_ppi(ppi, score='escore', threshold=0.0):
#     # Create a new directed graph from edge list
#     G = nx.from_pandas_edgelist(ppi, 'preferredName_A', 'preferredName_B', [score])

#     # Get positions for the nodes in G
#     pos = nx.spring_layout(G)

#     # Create Edges
#     edge_x = []
#     edge_y = []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])

#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines')

#     # Create Nodes
#     node_x = [pos[node][0] for node in G.nodes()]
#     node_y = [pos[node][1] for node in G.nodes()]

#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers',
#         hoverinfo='text',
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',
#             reversescale=True,
#             color=[],
#             size=10,
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line_width=2))

#     # Create Node Labels
#     node_labels = go.Scatter(
#         x=node_x,
#         y=node_y,
#         mode='text',
#         text=list(G.nodes()),
#         textposition='top center')

#     # Create Figure
#     fig = go.Figure(data=[edge_trace, node_trace, node_labels],
#                     layout=go.Layout(
#                         title=f'PPI Network (Score: {score}, Threshold: {threshold})',
#                         titlefont_size=16,
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=20, l=5, r=5, t=40),
#                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#                     )

#     fig.show()




def visualize_ppi(ppi, score='escore', threshold=0.0):
    # Create a new directed graph from edge list
    G = nx.from_pandas_edgelist(ppi, 'preferredName_A', 'preferredName_B', [score])

    # Get positions for the nodes in G
    pos = nx.spring_layout(G)

    # Create Edges
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create Nodes
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    # Compute average score for each node
    node_avg_scores = [sum(d[score] for n, d in G[node].items()) / len(G[node]) if G[node] else 0 for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_avg_scores,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Average Scores',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Create Node Labels
    node_labels = go.Scatter(
        x=node_x,
        y=node_y,
        mode='text',
        text=list(G.nodes()),
        textposition='top center')

    # Create Figure
    fig = go.Figure(data=[edge_trace, node_trace, node_labels],
                    layout=go.Layout(
                        title=f'PPI Network (Score: {score}, Threshold: {threshold})',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()


# all_genes =  ['ESR1', 'EFTUD2', 'HSPA8', 'STAU1', 'SHMT2', 'ACTB', 'GSK3B', 'YWHAB', 'UBXN6', 'PRKRA', 'BTRC', 'DDX23', 'SSR1', 'TUBA1C', 'SNIP1', 'SRSF5', 'ERBB2', 'MKI67', 'PGR', 'PLAU',
            # 'HNRNPU', 'STAU1', 'KDM1A', 'SERBP1', 'DHX9', 'EMC1', 'SSR1', 'PUM1', 'CLTC', 'PRKRA', 'KRR1', 'OCIAD1', 'CDC73', 'SLC2A1', 'HIF1A', 'PKM', 'CADM1', 'EPCAM', 'ALCAM', 'PTK7',
            # 'HNRNPL', 'HNRNPU', 'HNRNPA1', 'ZBTB2', 'SERBP1', 'RPL4', 'HNRNPK', 'HNRNPR', 'TFCP2', 'DHX9', 'RNF4', 'PUM1', 'ABCC1', 'CD44', 'ALCAM', 'ABCG2', 'ALDH1A1', 'ABCB1', 'EPCAM', 'PROM1']