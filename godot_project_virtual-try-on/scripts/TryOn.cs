using Godot;
using System;
using System.Diagnostics;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text;
using System.Linq; // Added for FirstOrDefault, Skip, and Take

public partial class TryOn : Control
{
	private TextureRect _videoFeed;
	private bool _isStreaming = true;
	private Godot.HttpRequest _httpRequest = new HttpRequest();
	private readonly System.Net.Http.HttpClient _httpClient = new System.Net.Http.HttpClient { Timeout = TimeSpan.FromSeconds(10) };
	private bool _isDisposing = false;

	public override void _Ready()
	{
		_videoFeed = GetNode<TextureRect>("VideoFeed");
		AddChild(_httpRequest);
		if (_videoFeed == null)
		{
			GD.PrintErr("Node VideoFeed tidak ditemukan!");
			return;
		}
		StartVideoFeed();
	}

	private async Task StartVideoFeed()
	{
		GD.Print("Mengambil stream video dari http://127.0.0.1:5000/video_feed ...");
		while (_isStreaming && !_isDisposing)
		{
			try
			{
				GD.Print("Mengirim HTTP request...");
				using var request = new HttpRequestMessage(HttpMethod.Get, "http://127.0.0.1:5000/video_feed");
				HttpResponseMessage response = null;
				try
				{
					response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
					GD.Print("Response diterima, status: " + response.StatusCode);
					response.EnsureSuccessStatusCode();
				}
				catch (HttpRequestException httpEx)
				{
					//GD.PrintErr($"HTTP request gagal: {httpEx.Message}, StatusCode: {httpEx.StatusCode}");
					await Task.Delay(1000); // Tunggu sebelum retry
					continue;
				}
				catch (TaskCanceledException)
				{
					//GD.PrintErr("HTTP request timeout");
					await Task.Delay(1000);
					continue;
				}
				catch (Exception ex)
				{
					//GD.PrintErr($"Unexpected error saat HTTP request: {ex.Message}, StackTrace: {ex.StackTrace}");
					await Task.Delay(1000);
					continue;
				}

				// Verifikasi content type
				var contentType = response.Content.Headers.ContentType?.ToString();
				GD.Print("Content-Type: " + contentType);
				if (!contentType.Contains("multipart/x-mixed-replace"))
				{
					//GD.PrintErr($"Unexpected content type: {contentType}");
					response.Dispose();
					await Task.Delay(1000);
					continue;
				}

				// Dapatkan boundary
				string boundary = response.Content.Headers.ContentType.Parameters
					.FirstOrDefault(p => p.Name == "boundary")?.Value ?? "frame";
				boundary = "--" + boundary;
				GD.Print("Boundary detected: " + boundary);

				using var stream = await response.Content.ReadAsStreamAsync();
				using var memoryStream = new System.IO.MemoryStream();
				var buffer = new byte[8192];
				int consecutiveEmptyReads = 0;
				const int maxBufferSize = 1024 * 1024; // Batasi buffer hingga 1 MB

				while (_isStreaming && !_isDisposing)
				{
					int bytesRead = await stream.ReadAsync(buffer, 0, buffer.Length);
					if (bytesRead == 0)
					{
						consecutiveEmptyReads++;
						if (consecutiveEmptyReads > 5)
						{
							GD.PrintErr("Stream ended unexpectedly after multiple empty reads");
							break;
						}
						await Task.Delay(10);
						continue;
					}
					consecutiveEmptyReads = 0;

					if (memoryStream.Length + bytesRead > maxBufferSize)
					{
						GD.PrintErr("Memory stream exceeded max size, clearing buffer");
						memoryStream.SetLength(0);
					}

					await memoryStream.WriteAsync(buffer, 0, bytesRead);

					byte[] frameData = memoryStream.ToArray();
					string frameString = System.Text.Encoding.ASCII.GetString(frameData);
					int boundaryIndex = frameString.IndexOf(boundary, StringComparison.Ordinal);
					if (boundaryIndex == -1)
						continue;

					int headerEnd = frameString.IndexOf("\r\n\r\n", boundaryIndex) + 4;
					if (headerEnd < 4)
						continue;

					int nextBoundaryIndex = frameString.IndexOf(boundary, boundaryIndex + boundary.Length, StringComparison.Ordinal);
					int jpgEnd = nextBoundaryIndex > 0 ? nextBoundaryIndex : frameData.Length;

					byte[] jpgData = frameData.Skip(headerEnd).Take(jpgEnd - headerEnd).ToArray();
					if (jpgData.Length > 0)
					{
						// Debug: Simpan frame
						// System.IO.File.WriteAllBytes($"debug_frame_{DateTime.Now.Ticks}.jpg", jpgData);

						if (jpgData.Length >= 2 && jpgData[0] == 0xFF && jpgData[1] == 0xD8)
						{
							Image img = new Image();
							Error error = img.LoadJpgFromBuffer(jpgData);
							if (error != Error.Ok)
							{
								//GD.PrintErr($"Gagal memuat JPG: {error}, Data length: {jpgData.Length}");
								//GD.Print("First 10 bytes: " + BitConverter.ToString(jpgData.Take(10).ToArray()));
								continue;
							}

							Callable.From(() =>
							{
								if (_videoFeed != null && !_isDisposing)
								{
									ImageTexture tex = ImageTexture.CreateFromImage(img);
									_videoFeed.Texture = tex;
								}
							}).CallDeferred();
						}
						else
						{
							//GD.PrintErr($"Invalid JPEG header, Data length: {jpgData.Length}");
							//GD.Print("First 10 bytes: " + BitConverter.ToString(jpgData.Take(10).ToArray()));
						}
					}

					if (nextBoundaryIndex > 0)
					{
						byte[] remainingData = frameData.Skip(nextBoundaryIndex).ToArray();
						memoryStream.SetLength(0);
						memoryStream.Write(remainingData, 0, remainingData.Length);
					}
					else
					{
						memoryStream.SetLength(0);
					}

					await Task.Delay(1); // Cegah CPU overload
				}

				response.Dispose();
			}
			catch (Exception e)
			{
				//GD.PrintErr($"Gagal ambil frame: {e.Message}, StackTrace: {e.StackTrace}");
				if (!_isStreaming)
					break;
			}

			await Task.Delay(33); // Target 30 fps
		}
	}

	private void OnStopPressed()
	{
		_isDisposing = true;
		GetTree().ChangeSceneToFile("res://scene/main_menu.tscn");
	}
	
	private void _on_BtnTipe1_pressed()
	{
		SendPostRequestTipe(1);
	}
	
	private void _on_BtnTipe2_pressed()
	{
		SendPostRequestTipe(2);
	}
	
	private void SendPostRequestTipe(int tipe){
		// URL of the FastAPI endpoint
		string url = "http://127.0.0.1:5000/update_filter";

		// JSON payload
		string jsonPayload = $"{{\"tipe\": {tipe}}}";

		// Headers for the POST request
		string[] headers = new string[] { "Content-Type: application/json" };

		// Send the POST request
		Error error = _httpRequest.Request(url, headers, Godot.HttpClient.Method.Post, jsonPayload);
		if (error != Error.Ok)
		{
			GD.Print("Error initiating HTTP request: ", error);
		}
	}
}
