# helpers.py 

# write quick helper function that just returns cluster and gene name for a given junction
def get_cluster(junc, junction_ids_conversion):
    simple_data_junc = junction_ids_conversion[junction_ids_conversion["junction_id_index"] == junc]
    return simple_data_junc["Cluster"].values[0]

def get_gene(junc, junction_ids_conversion):
    simple_data_junc = junction_ids_conversion[junction_ids_conversion["junction_id_index"] == junc]
    return simple_data_junc["gene_id"].values[0]

