python3 data/run_download_all.py \
  --query-list data/query_list.txt \
  --index-dir /data1021/xukaichen/data/Atlas \
  --output-dir /data1021/xukaichen/data/Atlas
  --resume

python3 data/check_and_redownload_h5ad.py \
  --query-list data/query_list.txt \
  --index-dir /data1021/xukaichen/data/Atlas \
  --output-dir /data1021/xukaichen/data/Atlas