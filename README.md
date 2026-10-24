# Pythagoras Engine: Form (Blender add-on)

Build polyhedral formations from modules, face by face, right in Blender's
Edit Mode. Pick a face, choose one of 57 modules (Platonic, Archimedean,
Johnson and Catalan solids, prisms, antiprisms and the rhombic family), place
it. Then shape the formation with Blender's own tools, divide or carve its
faces, rig it with a skeleton made from its modules, pose it, and keep it in
your asset library.

- **Install:** in Blender 4.2 or later, Edit > Preferences > Get Extensions,
  search for **Pythagoras Engine: Form**.
- **Start:** in the 3D Viewport, Add > PE Formation and pick a start face;
  press **N** and open the **PE** tab.

## Reporting an issue

Open an issue on the [Issues page](https://github.com/soboof/pe-form-blender/issues).
It helps most to include:

- the Blender version and the add-on version (Edit > Preferences > Add-ons);
- your operating system;
- what you did, step by step, what you expected, and what happened instead;
- any error text from Blender's Info area or the system console;
- if you can, the formation itself: the `.blend` (File > Save), or the
  formation alone (**Save formation (.json)** in the PE tab), attached to the
  issue.

## Source

`pe_form/` is the add-on exactly as it ships on the Blender Extensions
platform. `pe_form/engine/` is the placement engine in pure Python (no `bpy`);
`pe_form/bl/` is Blender's side of it: the operators, the panels, the mesh,
the skeleton and the asset library.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
