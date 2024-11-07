# OpenSearch 2.11
docker pull opensearchproject/opensearch:2.11.0

# OpenSearch Dashboard 2.11
docker pull opensearchproject/opensearch-dashboards:2.11.0

docker run -d --name opensearch \
  -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.11.0

# default OpenSearch memory allocation:
# -Xms2g (initial Heap size) -Xmx2g (maximum heapsize)

docker run -d --name opensearch-dashboards \
  -p 5601:5601 \
  --link opensearch:opensearch \
  -e "OPENSEARCH_HOSTS=http://opensearch:9200" \
  -e "DISABLE_SECURITY_DASHBOARDS_PLUGIN=true" \
  opensearchproject/opensearch-dashboards:2.11.0


# Wait a few seconds for OpenSearch to start
sleep 30

# Enable hybrid search plugin
curl -XPUT "http://localhost:9200/_search/pipeline/nlp-search-pipeline" -H 'Content-Type: application/json' -d'
{
  "description": "Post processor for hybrid search",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [
              0.3,
              0.7
            ]
          }
        }
      }
    }
  ]
}
'