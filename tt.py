# import plotly.graph_objects as go
#
# # داده‌ها
# labels = ['موافق', 'مخالف']
# sizes_eTOM = [75, 25]  # درصد موافقت و مخالفت با eTOM
# sizes_OSS = [80, 20]  # درصد موافقت و مخالفت با OSS
#
# # رسم نمودار
# fig = go.Figure()
# fig.add_trace(go.Pie(labels=labels, values=sizes_eTOM, name='eTOM', marker_colors=['#1f77b4', '#ff7f0e']))
# fig.add_trace(go.Pie(labels=labels, values=sizes_OSS, name='OSS', marker_colors=['#1f77b4', '#ff7f0e']))
#
# # تنظیمات دیگر
# fig.update_layout(title='توزیع موافقت کنندگان با eTOM و OSS')
#
# # نمایش نمودار
# fig.show()


import plotly.graph_objects as go

# داده‌ها
years = ['قبل', 'بعد']
productivity_eTOM = [60, 95]  # تغییرات بهره‌وری شبکه با انطباق eTOM
productivity_OSS = [55, 95]  # تغییرات بهره‌وری شبکه با استفاده از OSS

# رسم نمودار
fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=productivity_eTOM, mode='lines+markers', name='eTOM'))
fig.add_trace(go.Scatter(x=years, y=productivity_OSS, mode='lines+markers', name='OSS'))

# تنظیمات دیگر
fig.update_layout(title='تغییرات بهره‌وری شبکه‌های مخابراتی قبل و بعد از انطباق با eTOM و استفاده از OSS',
                  xaxis=dict(title='سال'),
                  yaxis=dict(title='درصد بهره‌وری'))

# نمایش نمودار
fig.show()
