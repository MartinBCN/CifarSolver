
import dash_html_components as html
import dash_core_components as dcc

from dashboard.maindash import app
# from tabs.map import generate_map
#
# from tabs.cases import generate_cases
# from tabs.scatter_plots import generate_scatter
from dashboard.tabs.tab_random_image import build_random_image


def build_banner():
    return html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="plotly-image",
                            style={
                                "height": "60px",
                                "width": "auto",
                                "margin-bottom": "25px",
                            },
                        )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Cifar10 Classifier",
                                    style={"margin-bottom": "0px"},
                                ),
                                html.H5(
                                    "Analysis", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-third column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Learn More", id="learn-more-button"),
                            href="https://github.com/MartinBCN/CoronaDashboard",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        )


def build_tabs():
    tab_confusion = dcc.Tab(
                        id="tab-confusion",
                        label="Confusion Matrix",
                        value="tab_confusion",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=html.Div()
    )

    tab_random_image = dcc.Tab(
                        id="tab-random-image",
                        label="Random Images",
                        value="tab_random_image",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=build_random_image()
                    )

    tabs = html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="tabs-cifar",
                value="tab_confusion",  # Default Choice
                className="custom-tabs",
                children=[tab_confusion, tab_random_image],
            )
        ],
    )

    return tabs


def main_layout():
    return html.Div(children=[
                html.Div(id="output-clientside"),
                build_banner(),
                build_tabs()
            ],
                id="mainContainer",
                style={"display": "flex", "flex-direction": "column"})
