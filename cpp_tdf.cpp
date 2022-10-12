// cppimport
#include <pybind11/pybind11.h>



PYBIND11_MODULE(cpp_tdf, m) {
    m.doc() = "This is a Python binding of a Two Dimensional Polynomial Least Squares Fit";

    // py::class_<CMM_Sheave>(m, "CMM_Sheave")
    //     .def(py::init<double, double, double, double, double, double, bool, bool>())
    //     .def(py::init<double, double, double, double, double, double, bool>())
    //     .def(py::init<double, double, double, double, double, double>())

    //     .def("psi", &CMM_Sheave::psi)
    //     .def("set_theta_c", &CMM_Sheave::setThetaC)
    //     .def("kappa_func_init", &CMM_Sheave::kappaFuncInit)
    //     .def("kappa_slope", &CMM_Sheave::kappaSlope)
    //     .def("kappa", &CMM_Sheave::kappa)
    //     .def("press", &CMM_Sheave::press)
    //     .def("v_radial", &CMM_Sheave::vRadial)
    //     .def("v_tan", &CMM_Sheave::vTangent)
    //     .def("beta", &CMM_Sheave::beta)
    //     .def("beta_s", &CMM_Sheave::betaS)
    //     .def("dimless_clamp", &CMM_Sheave::dimlessClamp)
    //     .def("force_ratio", &CMM_Sheave::forceRatio)
    //     .def("c", &CMM_Sheave::c)

    //     .def_static("kappaToTension", &CMM_Sheave::kappaToTension)
    //     .def_static("sToAxialClamp", &CMM_Sheave::sToAxialClamp)
    //     .def_static("axialClampToS", &CMM_Sheave::axialClampToS)
    //     .def_static("cToTorque", &CMM_Sheave::cToTorque)
    //     .def_static("torqueToC", &CMM_Sheave::torqueToC)
    //     .def_static("forceRatioToBeltTensions", &CMM_Sheave::forceRatioToBeltTensions)
    //     .def_static("beltTensionsToForceRatio", &CMM_Sheave::beltTensionsToForceRatio)
    //     .def_static("dLdRp", &CMM_Sheave::dLdRp)
    //     .def_static("dLdRs", &CMM_Sheave::dLdRs);
}


/*
<%
setup_pybind11(cfg)
%>
*/
