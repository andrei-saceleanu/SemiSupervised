import plotly.graph_objects as go



# fig = go.Figure()
# fig.add_trace(go.Bar(x=list(map(str, [3, 10, 50])),y=[64.8, 62.1, 56.3]))

# fig.update_layout(
#     title="SSL",
#     xaxis_title="Gamma parameter",
#     yaxis_title="Test accuracy",
#     yaxis_range=[50,66],
#     # legend_title="Legend",
#     # showlegend=True,
#     font=dict(
#         family="Courier New, monospace",
#         size=30,
#         color="RebeccaPurple"
#     )
# )
# fig.show()
# all methods on same graph

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3,4],y = [70.2,73.5,77.2,80.3],mode="lines+markers",name="FixMatch"))
fig.add_trace(go.Scatter(x=[1,2,3,4],y = [66.5,71.1,74.2,77.8],mode="lines+markers",name="FreeMatch"))
fig.add_trace(go.Scatter(x=[1,2,3,4],y = [64.1,72.3,76.7,77.6],mode="lines+markers",name="FixMatch+CR"))
# fig.add_trace(go.Scatter(x=[1,2,3,4],y = [55.7,60.9,66.3,68.5],mode="lines+markers",name="MeanTeacher"))

# fig.add_trace(go.Scatter(x=[1,2,3,4],y = [54.7,65.7,66.9,68.4],mode="lines+markers",name="NoisyStudent"))
fig.add_trace(go.Scatter(x=[1,2,3,4],y = [68.8,74.2,77.1,78.4],mode="lines+markers",name="MixMatch"))
fig.add_trace(go.Scatter(x=[1,2,3,4],y = [64.8,72.8,76.7,78.8],mode="lines+markers",name="LabelProp"))

# fig.add_hline(y=81, line_dash="dot",
#               annotation_text="Supervised 100%", 
#               annotation_position="top right",
#               annotation_font_size=20,
#               annotation_font_color="black"
#              )
# fig.add_hline(y=60.8, line_dash="dot",
#               annotation_text="Supervised 5%", 
#               annotation_position="top right",
#               annotation_font_size=25,
#               annotation_font_color="black"
#              )
# fig.add_hline(y=50.2, line_dash="dot",
#               annotation_text="Supervised 5%(frozen)", 
#               annotation_position="top right",
#               annotation_font_size=25,
#               annotation_font_color="black"
#              )
fig.update_xaxes(
    tickvals=[1,2,3,4],
    ticktext=["5%","20%","60%","80%"]
)
fig.update_layout(
    title="SSL",
    xaxis_title="Label percent",
    yaxis_title="Test accuracy",
    # yaxis_range=[47,82],
    legend_title="Legend",
    showlegend=True,
    font=dict(
        family="Courier New, monospace",
        size=30,
        color="RebeccaPurple"
    )
)
fig.show()