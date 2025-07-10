# 全局配置
COLUMN_NAMES = [
    'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
]

FEATURE_COLS = [
    'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Sub_metering_remainder',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU', 'day_of_week', 'month', 'day_of_year'
]

SHORT_TERM_STEPS = 90
LONG_TERM_STEPS = 365
TARGET_COL = 'Global_active_power'
NUM_EXPERIMENTS = 5