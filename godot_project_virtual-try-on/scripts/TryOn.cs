using Godot;
using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text;
using System.Text.Json;

public partial class TryOn : Control
{
	private TextureRect _videoFeed;
	private OptionButton _maskDropdown;
	private bool _isStreaming = true;
	private Godot.HttpRequest _httpRequest;
	private readonly System.Net.Http.HttpClient _httpClient = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(10) };
	private bool _isDisposing = false;
	private double _frameTimer = 0;
	private const double FrameInterval = 0.033; // ~30 FPS

	public override void _Ready()
	{
		_videoFeed = GetNode<TextureRect>("VideoFeed");
		_maskDropdown = GetNode<OptionButton>("MaskDropdown");
		
		// Create HttpRequest for frame fetching
		_httpRequest = new Godot.HttpRequest();
		AddChild(_httpRequest);
		_httpRequest.RequestCompleted += OnFrameReceived;
		
		if (_videoFeed == null)
		{
			GD.PrintErr("Node VideoFeed tidak ditemukan!");
			return;
		}
		
		if (_maskDropdown == null)
		{
			GD.PrintErr("Node MaskDropdown tidak ditemukan!");
			return;
		}
		
		// Load available masks dari backend
		LoadAvailableMasks();
	}

	public override void _Process(double delta)
	{
		if (!_isStreaming || _isDisposing)
			return;
			
		_frameTimer += delta;
		if (_frameTimer >= FrameInterval)
		{
			_frameTimer = 0;
			RequestFrame();
		}
	}

	private void RequestFrame()
	{
		// Check if already requesting to avoid multiple concurrent requests
		if (_httpRequest.GetHttpClientStatus() != Godot.HttpClient.Status.Disconnected)
			return; // Skip if still processing previous request
			
		Error err = _httpRequest.Request("http://127.0.0.1:5000/get_frame");
		if (err != Error.Ok)
		{
			// Silently skip, will retry on next frame
		}
	}

	private void OnFrameReceived(long result, long responseCode, string[] headers, byte[] body)
	{
		if (responseCode != 200 || body.Length == 0)
			return;
			
		try
		{
			// Validate JPEG header
			if (body.Length < 2 || body[0] != 0xFF || body[1] != 0xD8)
				return;
				
			// Validate JPEG footer
			if (body[body.Length - 2] != 0xFF || body[body.Length - 1] != 0xD9)
				return;
			
			Image img = new Image();
			Error error = img.LoadJpgFromBuffer(body);
			
			if (error == Error.Ok && _videoFeed != null && !_isDisposing)
			{
				ImageTexture tex = ImageTexture.CreateFromImage(img);
				_videoFeed.Texture = tex;
			}
		}
		catch
		{
			// Silently skip invalid frames
		}
	}

	private async void LoadAvailableMasks()
	{
		try
		{
			var response = await _httpClient.GetStringAsync("http://127.0.0.1:5000/available_masks");
			var jsonDoc = JsonDocument.Parse(response);
			var masks = jsonDoc.RootElement.GetProperty("masks");
			
			_maskDropdown.Clear();
			int index = 0;
			foreach (var mask in masks.EnumerateArray())
			{
				string maskName = mask.GetString();
				_maskDropdown.AddItem(maskName, index++);
			}
			
			GD.Print($"Loaded {index} masks");
		}
		catch (Exception e)
		{
			GD.PrintErr($"Gagal load masks: {e.Message}");
			// Fallback: tambah mask default
			_maskDropdown.AddItem("filter", 0);
		}
	}

	private async void OnMaskSelected(int index)
	{
		string selectedMask = _maskDropdown.GetItemText(index);
		GD.Print($"Mask dipilih: {selectedMask}");
		
		try
		{
			var payload = new { mask_name = selectedMask };
			var json = JsonSerializer.Serialize(payload);
			var content = new StringContent(json, Encoding.UTF8, "application/json");
			
			var response = await _httpClient.PostAsync("http://127.0.0.1:5000/select_mask", content);
			var result = await response.Content.ReadAsStringAsync();
			GD.Print($"Response: {result}");
		}
		catch (Exception e)
		{
			GD.PrintErr($"Gagal select mask: {e.Message}");
		}
	}

	private void OnStopPressed()
	{
		_isDisposing = true;
		_isStreaming = false;
		GetTree().ChangeSceneToFile("res://scene/main_menu.tscn");
	}
	
	private void _on_BtnTipe1_pressed()
	{
		// Untuk kompatibilitas dengan button lama, bisa dihapus jika tidak digunakan
		GD.Print("Filter 1 pressed - gunakan dropdown sebagai gantinya");
	}
	
	private void _on_BtnTipe2_pressed()
	{
		// Untuk kompatibilitas dengan button lama, bisa dihapus jika tidak digunakan
		GD.Print("Filter 2 pressed - gunakan dropdown sebagai gantinya");
	}
}
