import os
import tkinter
import customtkinter
from tkintermapview import TkinterMapView
import sys
import Side_menu
from dataloader import fishingDataLoader
import asyncio
import threading


customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green


class App(customtkinter.CTk):
    APP_NAME = "Gui"
    WIDTH = 1500
    HEIGHT = 1200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



        x_pos = int((self.winfo_screenwidth() / 2) - (App.WIDTH / 2))
        y_pos = int((self.winfo_screenheight() / 2) - (App.WIDTH / 2))

        self.title(App.APP_NAME)
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}+{x_pos}+{y_pos}")
        self.minsize(App.WIDTH, App.HEIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.bind("<Command-q>", self.on_closing)
        self.bind("<Command-w>", self.on_closing)
        self.createcommand('tk::mac::Quit', self.on_closing)

        # ============ create two CTkFrames ============

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_right = customtkinter.CTkFrame(master=self, corner_radius=0)
        self.frame_right.grid(row=0, column=0, rowspan=1, columnspan=2, pady=0, padx=0, sticky="nsew")

        self.frame_left = customtkinter.CTkFrame(master=self, width=150, corner_radius=20)
        self.frame_left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # ============ frame_left ============

        self.frame_left.grid_rowconfigure(2, weight=1)

        self.sidebar = Side_menu.ScrollableSideMenue(master=self.frame_left,
                                                     corner_radius=20,
                                                     bg_color='white')

        #self.testLoad()

        self.sidebar.grid(row=0, column=1, padx=0, pady=0, sticky="nesw")
        # ============ frame_right ============

        self.frame_right.grid_rowconfigure(1, weight=1)
        self.frame_right.grid_rowconfigure(0, weight=0)
        self.frame_right.grid_columnconfigure(0, weight=1)
        self.frame_right.grid_columnconfigure(1, weight=0)
        self.frame_right.grid_columnconfigure(2, weight=1)

        self.map_widget = TkinterMapView(self.frame_right, corner_radius=0)
        self.map_widget.grid(row=0, rowspan=2, column=0, columnspan=3, sticky="nswe", padx=(0, 0), pady=(0, 0))


        """self.search_frame = customtkinter.CTkFrame(master=self.frame_right, width=500, corner_radius=20)
        self.search_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.entry = customtkinter.CTkEntry(master=self.search_frame,
                                            width=300,
                                            placeholder_text="type address")

        self.entry.grid(row=0, column=1, padx=20, pady=12, sticky="e")
        self.entry.bind("<Return>", self.search_event)

        self.button_5 = customtkinter.CTkButton(master=self.search_frame,
                                                text="Search",
                                                width=90,
                                                command=self.search_event)
        self.button_5.grid(row=0, column=1, sticky="w", padx=12, pady=12)"""

        # Set default values
        self.map_widget.set_address("Wismar")

    def loadData(self):
        print("start async")
        loader = fishingDataLoader(path="../data/data")
        raw_data = loader.loadAllTrainingData()
        self.sidebar.loadListPandasGroups(raw_data.groupby("mmsi"))
        print("finished async")

    def testLoad(self):
        threading.Thread(target=self.loadData).start()

    def search_event(self, event=None):
        self.map_widget.set_address(self.entry.get())

    def change_appearance_mode(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_map(self, new_map: str):
        if new_map == "OpenStreetMap":
            self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
        elif new_map == "Google normal":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga",
                                            max_zoom=22)
        elif new_map == "Google satellite":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}&s=Ga",
                                            max_zoom=22)

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
