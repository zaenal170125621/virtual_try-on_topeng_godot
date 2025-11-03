using Godot;
using System;

public partial class Credit : Control
{
	private void _on_BtnBack_pressed()
	{
		GetTree().ChangeSceneToFile("res://scene/main_menu.tscn");
	}
}
