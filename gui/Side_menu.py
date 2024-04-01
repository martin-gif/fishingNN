import customtkinter
import pandas as pd


class ScrollableSideMenue(customtkinter.CTkScrollableFrame):
    def __init__(self, master, item_list=None, command=None, **kwargs):
        super().__init__(master, **kwargs)

        print(super().winfo_height())

        self.command = command
        self.entrys = []

        if item_list is not None:
            self.loadList(item_list)

    def loadList(self, itemlist):
        ...

    def loadListPandasGroups(self, itemlist):
        for i,(name,group) in enumerate(itemlist):
            button = customtkinter.CTkButton(self,
                                             text=name[:-2],
                                             height=40,
                                             width=180)
            button.grid(row=i,column=0, padx=10, pady=5, stick="w")
            self.entrys.append(button)

    def get(self):
        ...