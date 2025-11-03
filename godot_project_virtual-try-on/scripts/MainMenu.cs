using Godot;
using System;

public partial class MainMenu : Control
{
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
	}

	private void _on_BtnStart_pressed()
	{
		GetTree().ChangeSceneToFile("res://scene/try_on.tscn");
	}
	
	private void _on_BtnCredit_pressed()
	{
		GetTree().ChangeSceneToFile("res://scene/credit.tscn");
	}
	
		private void _on_BtnGuide_pressed()
	{
		GetTree().ChangeSceneToFile("res://scene/guide.tscn");
	}
	
	private void _on_BtnExit_pressed()
	{
		GetTree().Quit();
	}
}
