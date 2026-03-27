# 常见问题 Runbook

## Redis 连接超时

**症状**: `redis.exceptions.ConnectionError: Connection timed out`

**常见原因**:
1. 连接池 `max_connections` 不足 — 检查 QPS 与连接池大小的比例
2. Redis 实例内存满 — 检查 `used_memory` 和 `maxmemory`
3. 网络分区 — 检查 pod 和 redis 实例之间的网络连通性

**处理步骤**:
1. 确认告警时间范围和影响服务
2. 查看 redis 监控面板（连接数、内存、QPS）
3. 检查服务端连接池配置
4. 如果是连接池不足：临时调大 `REDIS_MAX_CONN` 环境变量并滚动重启

---

## MySQL 慢查询

**症状**: 接口 P99 延迟飙升，数据库 CPU 告警

**常见原因**:
1. 缺少索引 — `EXPLAIN` 查看执行计划
2. 全表扫描 — 检查 WHERE 条件
3. 锁等待 — 检查 `SHOW PROCESSLIST`

**处理步骤**:
1. 从慢查询日志找到 TOP SQL
2. EXPLAIN 分析执行计划
3. 临时方案：KILL 阻塞查询
4. 长期方案：添加索引或优化 SQL
