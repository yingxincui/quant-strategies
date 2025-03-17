import backtrader as bt
from loguru import logger

class TrailingStop(bt.Indicator):
    """
    追踪止损指标
    当价格上涨时，止损线会跟随上涨，但不会随价格下跌而下跌
    """
    lines = ('trailing_stop',)  # 声明指标线
    params = (
        ('trailing', 0.02),  # 追踪止损比例，默认2%
    )
    
    plotinfo = dict(subplot=False)  # 绘制在主图上
    
    def __init__(self):
        super(TrailingStop, self).__init__()
        self.plotinfo.plotname = f'TrailStop ({self.p.trailing:.1%})'  # 图例名称
        self.max_price = float('-inf')  # 初始化最高价为负无穷
        self.reset_requested = False  # 用于标记是否需要重置
        self.in_trade = False  # 标记是否在交易中
        self.entry_price = None  # 记录入场价格
        self._prev_stop = 0.0  # 记录前一个止损价
        self._last_price = None  # 添加上次价格记录
        self._call_count = 0  # 添加调用计数
        
        logger.info("初始化追踪止损指标 - 止损比例: {:.1%}", self.p.trailing)
        
    def reset(self, price=None):
        """重置最高价和止损价"""
        # 使用传入的价格或当前收盘价
        self.entry_price = price if price is not None else self.data.close[0]
        self.max_price = self.entry_price
        self._prev_stop = self.max_price * (1.0 - self.p.trailing)
        self.lines.trailing_stop[0] = self._prev_stop
        self.reset_requested = False
        self.in_trade = True
        
    def next(self):
        self._call_count += 1
        current_price = self.data.close[0]
        self._last_price = current_price
        
        # 如果在交易中，更新最高价和止损价
        if self.in_trade:
            # 如果价格创新高
            if current_price > self.max_price:
                self.max_price = current_price
                new_stop = self.max_price * (1.0 - self.p.trailing)
                # 确保新的止损价不低于之前的止损价
                self._prev_stop = max(new_stop, self._prev_stop)
                self.lines.trailing_stop[0] = self._prev_stop
            else:
                # 保持之前的止损价格
                self.lines.trailing_stop[0] = self._prev_stop
        else:
            # 不在交易中，设置止损价为0
            self.lines.trailing_stop[0] = 0.0
            # 重置最高价，为下次交易做准备
            self.max_price = float('-inf')
            self._prev_stop = 0.0
            self.entry_price = None
        
    def stop_tracking(self):
        """停止追踪"""
        if self.in_trade:
            logger.info("停止追踪止损 - 入场价: {:.2f}, 最高价: {:.2f}, 最终止损价: {:.2f}", 
                       self.entry_price, self.max_price, self._prev_stop)
        self.in_trade = False 