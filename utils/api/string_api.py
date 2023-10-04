from io import StringIO
from time import sleep
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import requests


def get_ppi_encoder(chosen_genes: list[str], score: str = 'escore', threshold: float = 0.0):
    """Get PPI for chosen genes

    Args:
        chosen_genes (list[str]): list of chosen genes

    Returns:
        ppi (pd.DataFrame): PPI network
    """
    gene_encoder = dict(zip(chosen_genes, range(len(chosen_genes))))

    res_text = _get_ppi_network_from_string(chosen_genes)

    ppi = pd.read_csv(StringIO(res_text), sep='\t')
    ppi = ppi[['preferredName_A', 'preferredName_B', score]]
   
    
    #check if I used escore or score
    
    ppi.drop_duplicates(inplace=True)
    
    
    ppi = ppi.reset_index(drop=True)
    #filter by threshold
    #  
    ppi = ppi[ppi[score] >= threshold]
   
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

    def visualize_ppi(ppi):
        # Create a new directed graph from edge list
        G = nx.from_pandas_edgelist(ppi, 'src', 'dst', ['escore'])

        # You can choose different layouts for your graph
        pos = nx.spring_layout(G)

        # Draw nodes (color by degree)
        nodes = nx.draw_networkx_nodes(G, pos, node_color='blue')

        # Draw edges (color by weight)
        edges = nx.draw_networkx_edges(G, pos, edge_color='grey')

        # Draw labels
        labels = nx.draw_networkx_labels(G, pos)

        # Display
        plt.title('PPI Network')
        plt.show()


# all_genes =  ['ESR1', 'EFTUD2', 'HSPA8', 'STAU1', 'SHMT2', 'ACTB', 'GSK3B', 'YWHAB', 'UBXN6', 'PRKRA', 'BTRC', 'DDX23', 'SSR1', 'TUBA1C', 'SNIP1', 'SRSF5', 'ERBB2', 'MKI67', 'PGR', 'PLAU',
            'HNRNPU', 'STAU1', 'KDM1A', 'SERBP1', 'DHX9', 'EMC1', 'SSR1', 'PUM1', 'CLTC', 'PRKRA', 'KRR1', 'OCIAD1', 'CDC73', 'SLC2A1', 'HIF1A', 'PKM', 'CADM1', 'EPCAM', 'ALCAM', 'PTK7',
            'HNRNPL', 'HNRNPU', 'HNRNPA1', 'ZBTB2', 'SERBP1', 'RPL4', 'HNRNPK', 'HNRNPR', 'TFCP2', 'DHX9', 'RNF4', 'PUM1', 'ABCC1', 'CD44', 'ALCAM', 'ABCG2', 'ALDH1A1', 'ABCB1', 'EPCAM', 'PROM1']