# import plotly.graph_objects as go
#
# # داده‌ها
# categories = ['پوشش نیازمندی های سازمان', 'انطباق با فرآیندها', 'افزایش بهره‌وری شبکه‌ها']
# percentages = [75, 80, 35]
#
# # رسم نمودار
# fig = go.Figure()
#
# # میله‌ها
# fig.add_trace(go.Bar(
#     x=categories,
#     y=percentages,
#     marker=dict(color=['blue', 'green', 'orange']),
#     width=0.5,
#     opacity=0.6
# ))
#
# # اضافه کردن متن به بالای میله‌ها
# for i in range(len(categories)):
#     fig.add_annotation(
#         x=categories[i],
#         y=percentages[i],
#         text=f"{percentages[i]}%",
#         font=dict(color='blue', size=14),
#         showarrow=False,
#         xanchor='center',
#         yanchor='bottom'
#     )
#
# # تنظیمات دیگر
# fig.update_layout(
#     title='مقایسه درصد موافقت با eTOM و اثرات آن بر بهره‌وری شبکه‌های مخابراتی',
#     xaxis=dict(title='موارد'),
#     yaxis=dict(title='درصد'),
#     showlegend=False
# )
#
# # نمایش نمودار
# fig.show()

import plotly.graph_objects as go

# داده‌ها
categories = ['موافقت با OSS', 'انطباق با بهبود بهره‌وری', 'افزایش بهره‌وری شبکه‌ها با OSS']
percentages = [80, 85, 40]

# رسم نمودار
fig = go.Figure()

# میله‌ها
fig.add_trace(go.Bar(
    x=categories,
    y=percentages,
    marker=dict(color=['blue', 'green', 'orange']),
    width=0.5,
    opacity=0.6
))

# اضافه کردن متن به بالای میله‌ها
for i in range(len(categories)):
    fig.add_annotation(
        x=categories[i],
        y=percentages[i],
        text=f"{percentages[i]}%",
        font=dict(color='blue', size=14),
        showarrow=False,
        xanchor='center',
        yanchor='bottom'
    )

# تنظیمات دیگر
fig.update_layout(
    title='مقایسه درصد موافقت با OSS و اثرات آن بر بهره‌وری شبکه‌های مخابراتی',
    xaxis=dict(title='موارد'),
    yaxis=dict(title='درصد'),
    showlegend=False
)

# نمایش نمودار
fig.show()
