# Windows 每日任务

## What was added

任务注册脚本和执行脚本，将每日推荐 CLI 交给 Windows 任务计划程序。
默认北京时间 08:00，注册时读取用户配置的 `daily_time`。执行隐藏 PowerShell 窗口，写入本地日志。
机器错过触发时间后补跑当前日期，任务运行时不启动重叠实例。

## Why it exists

每日推荐不依赖 API 服务持续运行。任务使用项目的 `.venv` 和绝对路径，避免系统 Python 或工作目录差异。
本地电脑关机时无法执行，开机后的补跑不会生成过期清单。

## How to test or use it

初始化偏好、配置凭据并完成真实接入检查后，在项目根目录运行：

```powershell
.\scripts\Install-DailyTask.ps1 -WhatIf
.\scripts\Install-DailyTask.ps1
Get-ScheduledTask -TaskName VideoAgent-Daily
Start-ScheduledTask -TaskName VideoAgent-Daily
```

要求 Windows 时区为 `China Standard Time`。脚本不会改变系统时区，也不会覆盖同名任务。
注册后更改配置中的时间，需要在任务计划程序中同步更新触发器。
任务日志位于配置的数据目录 `daily-task.log`；API 查询对应日期可查看业务结果。

本次交付仅提供并静态检查脚本，未在缺少真实服务配置时注册实际每日任务。
