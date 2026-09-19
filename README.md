# Nifty 500 daily scan

GitHub Actions runs scanner.py at 06:30 IST, Monday to Friday. It scans the previous session's daily candle
for every Nifty 500 stock (candlestick patterns and technical setups), draws charts for the top signals
and publishes the report to reports/. A Claude scheduled task emails reports/latest.json -> html_url
through the Gmail connector at 08:00 IST. The repo must stay public so Gmail can load the chart images.

Manual run: Actions tab -> "Nifty 500 daily scan" -> Run workflow.
