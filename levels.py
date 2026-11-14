"""Support and resistance zones and swing levels at the last completed candle."""
import numpy as np
import pandas as pd


class LevelIndicator:
  """Support and resistance for one ladder: ``zones`` and ``swingLevels`` each take its candles and return what
  shows at the last candle given; for an earlier moment, pass the candles up to it.

  ``candles`` is a DataFrame of completed candles, oldest first, with columns ``time`` (the candle's open, echoed
  back as pivot times), ``session`` (one value per exchange session, changing at the next), ``high``, ``low``,
  ``close`` and optionally ``volume``. Zones are meant for 5-minute candles; swing levels use whatever timeframe
  is given. Sessions are counted from the ``session`` values present, so one without candles does not count.

  ``structureCode`` is the structure's leg coefficients (``'1'`` outright, ``'1,-1'`` adjacent calendar spread);
  any code not in ``SATURATION_BY_CODE`` scores with ``DEFAULT_SATURATION``.
  """

  # Sessions before the current one that zones are found in.
  ZONE_SESSIONS = 100
  # Candles either side that a zone pivot's high must top, or its low must bottom.
  ZONE_PIVOT_BARS = 72
  # A pivot at the last counted pivot's side and exact price, within this many candles of it, is not counted again.
  ZONE_SUPPRESSION_BARS = 72
  # Pivots a band needs to be a zone.
  ZONE_TOUCHES = 2
  # Candles in the ATR that sets a band's width.
  ZONE_ATR_PERIOD = 20
  # A band spans at most twice this many band ATRs.
  ZONE_HALF_WIDTH_ATR = 0.5
  # Sessions in the daily ATR that distance from price is measured in.
  ZONE_DAILY_ATR_PERIOD = 20
  # A zone is in play while the last close is within this many daily ATRs of its nearer edge.
  ZONE_IN_PLAY_DAILY_ATR = 1
  # Zones returned at most, best score first.
  ZONE_MAX = 3
  # Candles before a swing pivot that it must top or bottom, and candles after it that it must survive to confirm.
  SWING_PIVOT_BARS = 20
  # Swing pivots come from the longer of this many sessions, the current one included, and this many candles.
  SWING_HISTORY_SESSIONS = 10
  SWING_HISTORY_BARS = 60
  # The share of swing levels, longest standing first, that may be returned.
  SWING_TOP_PERCENT = 50
  # Swing levels returned each side of the last close.
  SWING_PER_SIDE = 1
  # A zone's score is 1 - exp(-touches / saturation), by structure code.
  SATURATION_BY_CODE = {'1': 7.163, '1,-1': 6.038}
  DEFAULT_SATURATION = 3.0

  def __init__(self, structureCode='1'):
    self.structureCode = structureCode

  @staticmethod
  def _simpleAtr(high, low, close, period):
    previous = np.r_[np.nan, close[:-1]]
    ranges = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(previous - low)])
    valid = np.arange(len(close)) >= period
    sums = np.r_[0.0, 0.0, np.cumsum(ranges[1:])]
    ends = np.flatnonzero(valid)
    values = np.zeros(len(close))
    values[ends] = (sums[ends + 1] - sums[ends + 1 - period]) / period
    return values, valid

  @staticmethod
  def _extremes(values, radius, maximum=True):
    empty = -np.inf if maximum else np.inf
    padded = np.r_[np.full(radius, empty), values, np.full(radius, empty)]
    windows = np.lib.stride_tricks.sliding_window_view(padded, radius)
    extremes = (np.max if maximum else np.min)(windows, axis=1)
    return extremes[:len(values)], extremes[radius + 1:radius + 1 + len(values)]

  def zones(self, candles):
    """The zones in play at the last candle, best score first then lowest band first. ``touches`` counts the
    pivots in the band, ``score`` runs 0 to 1, ``widthAtr`` is in band ATRs, ``extremity`` runs 0 at the middle of
    the window's range to 1 at its high or low, ``side`` is the zone's against the last close. ``bandAtr`` and
    ``dailyAtr`` are in price units; ``dailyAtr`` is None with fewer than ``ZONE_DAILY_ATR_PERIOD + 1`` sessions,
    and distance is then measured in band ATRs. Zones are found in the sessions before the last candle's, never
    in its own, so they do not move during a session.
    """
    columns = ['low', 'high', 'mid', 'touches', 'score', 'widthAtr', 'extremity', 'side',
      'firstPivotTime', 'lastPivotTime', 'bandAtr', 'dailyAtr']
    if candles.empty:
      return pd.DataFrame(columns=columns)
    sessions, opens = pd.factorize(candles['session'], sort=False, use_na_sentinel=False)
    current = len(opens) - 1
    firstSession = max(0, current - self.ZONE_SESSIONS)
    keep = (sessions >= max(0, firstSession - 1)) & (sessions < current)
    if 'volume' in candles:
      keep &= candles['volume'].to_numpy() > 0
    high, low, close = (candles[name].to_numpy(dtype=float)[keep] for name in ('high', 'low', 'close'))
    times, sessions = candles['time'].to_numpy()[keep], sessions[keep]
    atrSums, atrCounts = [], []
    for session in range(firstSession, current):
      start = np.searchsorted(sessions, session, side='left')
      end = np.searchsorted(sessions, session, side='right')
      if end <= start:
        atrSums.append(0.0)
        atrCounts.append(0)
        continue
      first = max(0, start - self.ZONE_ATR_PERIOD)
      atr, valid = self._simpleAtr(high[first:end], low[first:end], close[first:end], self.ZONE_ATR_PERIOD)
      atrSums.append(float(atr[start - first:][valid[start - first:]].sum()))
      atrCounts.append(int(valid[start - first:].sum()))
    bandAtr = float(np.asarray(atrSums).sum() / sum(atrCounts)) if sum(atrCounts) else 0.0
    dailyAtr = None
    if len(sessions):
      starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
      ends = np.r_[starts[1:], len(sessions)] - 1
      atr, valid = self._simpleAtr(np.maximum.reduceat(high, starts), np.minimum.reduceat(low, starts),
        close[ends], self.ZONE_DAILY_ATR_PERIOD)
      if valid[-1] and atr[-1] > 0:
        dailyAtr = float(atr[-1])
    window = sessions >= firstSession
    high, low, times = high[window], low[window], times[window]
    if not len(high) or bandAtr <= 0:
      return pd.DataFrame(columns=columns)
    leftHigh, rightHigh = self._extremes(high, self.ZONE_PIVOT_BARS)
    leftLow, rightLow = self._extremes(low, self.ZONE_PIVOT_BARS, False)
    inner = (np.arange(len(high)) > 0) & (np.arange(len(high)) < len(high) - 1)
    isHigh = inner & (high >= leftHigh) & (high >= rightHigh)
    isLow = inner & (low <= leftLow) & (low <= rightLow)
    keptHigh, keptLow, anchor = [], [], None
    for row in np.flatnonzero(isHigh ^ isLow):
      side, price = ('high', high[row]) if isHigh[row] else ('low', low[row])
      if anchor is not None and anchor[:2] == (side, price) and row - anchor[2] <= self.ZONE_SUPPRESSION_BARS:
        continue
      (keptHigh if isHigh[row] else keptLow).append(row)
      anchor = (side, price, row)
    prices, pivotTimes = np.r_[high[keptHigh], low[keptLow]], np.r_[times[keptHigh], times[keptLow]]
    order = np.argsort(prices, kind='stable')
    prices, pivotTimes = prices[order], pivotTimes[order]
    width = 2 * self.ZONE_HALF_WIDTH_ATR * bandAtr
    saturation = self.SATURATION_BY_CODE.get(self.structureCode, self.DEFAULT_SATURATION)
    lastClose, span = candles['close'].iloc[-1], float(high.max()) - float(low.min())
    unit = bandAtr if dailyAtr is None else dailyAtr
    rows, start = [], 0
    while start < len(prices):
      end = int(np.searchsorted(prices, prices[start] + width, 'right'))
      bandLow, bandHigh, touches = prices[start], prices[end - 1], end - start
      distance = max(bandLow - lastClose, lastClose - bandHigh, 0) / unit
      if touches >= self.ZONE_TOUCHES and distance <= self.ZONE_IN_PLAY_DAILY_ATR:
        mid = (bandLow + bandHigh) / 2
        rows.append([bandLow, bandHigh, mid, touches,
          np.round(np.clip(1 - np.exp(-touches / saturation), 0, 1), 2),
          np.round((bandHigh - bandLow) / bandAtr, 4),
          np.round(2 * abs((mid - low.min()) / span - 0.5), 4) if span > 0 else 0.0,
          'above' if bandLow > lastClose else 'below' if bandHigh < lastClose else 'inside',
          pivotTimes[start:end].min(), pivotTimes[start:end].max(), bandAtr, dailyAtr])
      start = end
    inPlay = pd.DataFrame(rows, columns=columns).sort_values('score', ascending=False, kind='stable')
    return inPlay.head(self.ZONE_MAX).reset_index(drop=True)

  def swingLevels(self, candles):
    """The swing levels shown at the last candle: nearest above the last close first, then one exactly at it,
    then nearest below. ``side`` is the pivot's, ``'high'`` or ``'low'``. ``confirmed`` means no candle traded
    through it in the ``SWING_PIVOT_BARS`` after it; one traded through sooner is no level at all. ``broken``
    means a later candle did, and a broken level stays eligible. ``holdBars`` counts candles from the pivot to
    its break, or to the last candle while unbroken.
    """
    columns = ['price', 'side', 'pivotTime', 'confirmed', 'broken', 'holdBars']
    count = len(candles)
    if count < 3:
      return pd.DataFrame(columns=columns)
    sessions, opens = pd.factorize(candles['session'], sort=False, use_na_sentinel=False)
    firstSession = len(opens) - self.SWING_HISTORY_SESSIONS
    sessionStart = np.searchsorted(sessions, firstSession) if firstSession >= 0 else 0
    historyStart = min(sessionStart, max(count - self.SWING_HISTORY_BARS, 0))
    high, low = (candles[name].to_numpy(dtype=float) for name in ('high', 'low'))
    times, lastClose = candles['time'].to_numpy(), candles['close'].iloc[-1]
    candidates = []
    for side, values in (('high', high), ('low', -low)):
      left, _ = self._extremes(values, self.SWING_PIVOT_BARS)
      for row in range(max(historyStart, self.SWING_PIVOT_BARS), count - 1):
        if values[row] < left[row] or values[row] < values[row + 1]:
          continue
        beyond = np.flatnonzero(values[row + 1:] > values[row])
        breakRow = row + 1 + int(beyond[0]) if len(beyond) else -1
        if 0 <= breakRow <= row + self.SWING_PIVOT_BARS:
          continue
        confirmed = row + self.SWING_PIVOT_BARS < count
        broken = confirmed and breakRow >= 0
        hold = (breakRow if broken else count - 1) - row
        candidates.append((row, side, high[row] if side == 'high' else low[row], confirmed, broken, hold))
    best = {}
    for candidate in sorted(candidates, key=lambda item: (item[0], item[1] == 'low')):
      price = candidate[2]
      if price not in best or candidate[5] > best[price][5]:
        best[price] = candidate
    if not best:
      return pd.DataFrame(columns=columns)
    position = max(1, -(-len(best) * int(self.SWING_TOP_PERCENT) // 100))
    cutoff = sorted((item[5] for item in best.values()), reverse=True)[position - 1]
    kept = [item for item in best.values() if item[5] >= cutoff]
    above = sorted((item for item in kept if item[2] > lastClose), key=lambda item: item[2])[:self.SWING_PER_SIDE]
    atClose = [item for item in kept if item[2] == lastClose]
    below = sorted((item for item in kept if item[2] < lastClose), key=lambda item: -item[2])[:self.SWING_PER_SIDE]
    return pd.DataFrame([[price, side, times[row], confirmed, broken, hold]
      for row, side, price, confirmed, broken, hold in above + atClose + below], columns=columns)
