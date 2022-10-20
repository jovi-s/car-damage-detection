import warnings

import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from flask import Flask
from PIL import Image
from src.inference import load_saved_model, predict, visualize
from torchvision.io import read_image

warnings.filterwarnings("ignore")

server = Flask(__name__)
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(
    __name__,
    title="Car Defection Detection",
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # external_stylesheets
    server=server,
)
app.config.suppress_callback_exceptions = True
# server = app.server

model = load_saved_model()
# Choose Image
images_list = [
    "11.jpg",
    "12.jpg",
    "28.jpg",
    "45.jpg",
    "60.jpg",
    "66.jpg",
    "67.jpg",
    "72.jpg",
]


app.layout = html.Div(
    [
        html.Div(
            children=[
                html.P(
                    children="🚗",
                    className="header-emoji",
                ),
                html.H1("Car Defect Detection", className="header-title"),
                html.P(
                    children="Automated visual inspection AI system that detects car damages.",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(children="Model:", className="menu-title"),
                dcc.Dropdown(
                    options=["Mask R-CNN", "MaskFormer"],
                    value="Mask R-CNN",
                    clearable=False,
                    id="model-name",
                ),
                html.Div(children="Choose an image:", className="menu-title"),
                dcc.Dropdown(
                    id="image-dropdown",
                    options=[{"label": i, "value": i} for i in images_list],
                    # initially display the first entry in the list
                    value=images_list[0],
                    clearable=False,
                ),
            ],
            className="menu",
        ),
        html.Div(
            children=[
                html.Div(id="image"),
            ],
            className="wrapper",
        ),
    ]
)


@app.callback(
    Output("image", "children"),
    Input("image-dropdown", "value"),
    Input("model-name", "value"),
)
def image_inference(image_name, model_name):
    model_name = model_name
    img_path = "data/test/" + image_name

    original_image = Image.open(img_path)
    img_int = read_image(str(img_path))

    fig_original = px.imshow(original_image, title="Original Image")

    prediction = predict(model, original_image)
    output = visualize(img_int, prediction)
    fig_output = px.imshow(output, title="Detected Damage")

    layout = html.Div(
        [
            html.Div(
                children=[
                    dcc.Graph(
                        id="original-image",
                        figure=fig_original,
                        style={
                            "width": "60vh",
                            "height": "60vh",
                            "display": "inline-block",
                        },
                    ),
                    dcc.Graph(
                        id="detect-image",
                        figure=fig_output,
                        style={
                            "width": "60vh",
                            "height": "60vh",
                            "display": "inline-block",
                        },
                    ),
                ],
                className="row",
            ),
            dcc.Markdown(
                f"""
               Maximum Mask Score: {round(prediction[0]["scores"].max().item()*100, 1)}%
            """,
                className="conf-score",
            ),
        ]
    )
    return layout


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)
