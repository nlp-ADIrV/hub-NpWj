import asyncio
from core import MasterAgent
from tasks.sample_tasks import sample_async_task, sample_sync_task, failing_task


async def demo_parallel():
    """演示：纯并行执行"""
    print("=" * 60)
    print("🚀 演示：纯并行执行")
    print("=" * 60)

    master = MasterAgent(pool_size=3, max_concurrent=5)

    tasks = {
        "数据爬取": (sample_async_task, ("爬取用户数据", 2.0), {}),
        "数据分析": (sample_async_task, ("分析销售数据", 1.5), {}),
        "报告生成": (sample_async_task, ("生成日报", 1.0), {}),
        "同步任务": (sample_sync_task, ("文件整理", 1.0), {}),
    }

    results = await master.run_parallel(tasks)

    for name, result in results.items():
        print(result)
    print()


async def demo_dag():
    """演示：带依赖的 DAG 执行"""
    print("=" * 60)
    print("🔗 演示：带依赖的 DAG 执行")
    print("=" * 60)

    master = MasterAgent(pool_size=3, max_concurrent=5)

    tasks = {
        "fetch_data": {
            "func": sample_async_task,
            "args": ("爬取数据", 2.0),
            "depends_on": [],
        },
        "send_notification": {
            "func": sample_async_task,
            "args": ("发送通知", 1.0),
            "depends_on": [],
        },
        "analyze_data": {
            "func": sample_async_task,
            "args": ("分析数据", 1.5),
            "depends_on": ["fetch_data"],
        },
        "generate_report": {
            "func": sample_async_task,
            "args": ("生成报告", 1.0),
            "depends_on": ["analyze_data", "send_notification"],
        },
        "will_fail": {
            "func": failing_task,
            "args": ("异常任务",),
            "depends_on": [],
        },
    }

    results = await master.run_with_dependencies(tasks)

    for name, result in results.items():
        print(result)
    print()


async def main():
    await demo_parallel()
    await demo_dag()


if __name__ == "__main__":
    asyncio.run(main())