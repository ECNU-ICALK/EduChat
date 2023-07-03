# Clean Tool
## 使用方法
- 将要同时去重的多个数据文件置于./clean_tool/data/$NAME文件夹下，其中$NAME是本次去重数据的文件夹名，文件夹下去重文件以.jsonl结尾，文件内容格式参考Openassistant的指令数据格式与对话数据格式
- cd到clean_tool文件夹中
- 运行 `python get_emb.py $NAME`
- 如果您打算使用GPU加速，请运行 `python main-gpu.py $NAME $BLOCKSIZE $GPUNUM`，其中`$BLOCKSIZE`代表去重分块个数（请保证`$BLOCKSIZE`为2^k），`$GPUNUM`代表GPU个数
- 使用CPU进行去重，请运行 `python main.py $NAME $BLOCKSIZE`，其中`$BLOCKSIZE`代表去重分块个数（请保证$BLOCKSIZE为2^k）
- 去重后数据将保存在./clean_tool/data/MIX_$NAME.jsonl中
