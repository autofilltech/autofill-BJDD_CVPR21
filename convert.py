from glob import glob
import os
import shutil

for path in glob("./training/colored_0/*"):
	_,tail = os.path.split(path)
	fn = tail.split(".")[0]
	dst = os.path.join("./val", fn)
	os.makedirs(dst)

	src0 = os.path.join("./testing/colored_0", tail)
	src1 = os.path.join("./testing/colored_1", tail)
	dst0 = os.path.join(dst, "hr0.png")
	dst1 = os.path.join(dst, "hr1.png")

	shutil.copy(src0, dst0)
	shutil.copy(src1, dst1)
	print("Copied", fn)
