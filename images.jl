### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 65e9721a-f97f-11ea-07d2-31ce5cb2cec5
begin
	import Pkg
	Pkg.add("Images")
	using Images
	philip = load("philip.jpg")
end

# ╔═╡ 206d5c4c-f97f-11ea-31f5-6d73f00e7a7f
url = "https://i.imgur.com/VGPeJ6s.jpg"

# ╔═╡ 5f5ebe96-f97f-11ea-154a-59821c8ddab3
download(url, "philip.jpg")

# ╔═╡ Cell order:
# ╠═206d5c4c-f97f-11ea-31f5-6d73f00e7a7f
# ╠═5f5ebe96-f97f-11ea-154a-59821c8ddab3
# ╠═65e9721a-f97f-11ea-07d2-31ce5cb2cec5
