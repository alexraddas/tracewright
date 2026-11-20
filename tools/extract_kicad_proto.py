"""Dev tool: extract the KiCad IPC API protobuf descriptors from an installed kicad-python
(kipy) package into a FileDescriptorSet that build.rs feeds to prost-build.

Only needed when updating to a new KiCad API version. Run with a Python that has kipy:

    python tools/extract_kicad_proto.py proto/kicad_api.fds.bin [--list]
"""
import importlib
import pkgutil
import sys

from google.protobuf import descriptor_pb2
import kipy.proto as kp

names = [m.name for m in pkgutil.walk_packages(kp.__path__, kp.__name__ + ".")
         if m.name.endswith("_pb2")]

# The pure-python protobuf runtime needs dependencies registered first; retry until stable.
pending = list(names)
while pending:
    progress = False
    for n in list(pending):
        try:
            importlib.import_module(n)
            pending.remove(n)
            progress = True
        except Exception:
            sys.modules.pop(n, None)
    if not progress:
        print("skipping (unresolvable deps): %s" % pending, file=sys.stderr)
        names = [n for n in names if n not in pending]
        break

fds = descriptor_pb2.FileDescriptorSet()
seen = set()


def add(fd):
    if fd.name in seen:
        return
    for dep in fd.dependencies:
        add(dep)
    seen.add(fd.name)
    p = descriptor_pb2.FileDescriptorProto()
    fd.CopyToProto(p)
    fds.file.append(p)


for n in names:
    add(sys.modules[n].DESCRIPTOR)

with open(sys.argv[1], "wb") as f:
    f.write(fds.SerializeToString())

for f in fds.file:
    print(f.name, f.package, len(f.message_type))
    if "--list" in sys.argv:
        for mt in f.message_type:
            print("   ", mt.name, [fl.name for fl in mt.field])
